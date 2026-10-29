"""Select registered model/data, seeded samples, and immutable run inputs."""
import json
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor

from data.dataset_registry import dataset_dict
from models.model_registry import model_dict
from reconciliation import reconciliation
from run_utils.seeding import seed_everything, stage_seed
from .common import digest, model_digest, tensor_digest, write_json


GENERATION_DEFAULTS = dict(seed=2026, public_count=100, servers=3,
                           relative_margin=.01, partitions=8, anchors=16)


def resolve_settings(args):
    """Old output directories retain their selected image and share settings."""
    manifest_path = args.output / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    old_args = {} if old is None else old.get('arguments', {})
    for name, default in GENERATION_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, old_args.get(name, default))
    args.model = args.model or (old or {}).get('model', 'lenet5')
    args.dataset = args.dataset or (old or {}).get('dataset', 'cifar10')
    if args.private_index is None and old:
        args.private_index = old.get('private_index', old_args.get('private_index'))
    return old


def load_datasets(name, root, download=False):
    factories = dict(cifar10=CIFAR10, mnist=MNIST)
    factory = factories[name]
    options = dict(root=str(root), download=download, transform=ToTensor())
    return factory(train=False, **options), factory(train=True, **options)


def select_indices(private_size, public_size, public_count, seed, private_index=None):
    if not 1 <= public_count <= public_size:
        raise ValueError('public_count exceeds the distinct public population')
    if private_size < 1:
        raise ValueError('Private population is empty')
    if private_index is None:
        rng = torch.Generator().manual_seed(stage_seed(seed, 'private_sample'))
        private_index = int(torch.randint(private_size, (), generator=rng))
    if not 0 <= private_index < private_size:
        raise ValueError('Private sample index is out of range')
    rng = torch.Generator().manual_seed(stage_seed(seed, 'public_samples'))
    public_indices = torch.randperm(public_size, generator=rng)[:public_count].tolist()
    return private_index, public_indices


def build_model(name, definition, image, seed, state=None):
    seed_everything(stage_seed(seed, 'model'))
    model_definition = model_dict[name]
    model = reconciliation(model_definition, definition, model_definition.model_builder())
    model.eval()
    with torch.no_grad():
        model(image)  # Materialize lazy parameters using the independent model stream.
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def prepare_inputs(args, old):
    definition = dataset_dict[args.dataset]
    private_data, public_data = load_datasets(args.dataset, args.data, args.download)
    private_index, public_indices = select_indices(len(private_data), len(public_data),
                            args.public_count, args.seed, args.private_index)
    if old and old.get('protocol') == 'sparse_disguise_v1':
        # Historical public selection used seed directly rather than stage_seed.
        rng = torch.Generator().manual_seed(args.seed)
        public_indices = torch.randperm(len(public_data), generator=rng)[:args.public_count].tolist()
    private, y = private_data[private_index]
    private, label = private[None], torch.tensor([int(y)])
    public = torch.stack([public_data[i][0] for i in public_indices])
    public_labels = [int(public_data[i][1]) for i in public_indices]
    expected_shape = (definition.input_channels, *definition.input_dimensions)
    if tuple(private.shape[1:]) != expected_shape or tuple(public.shape[1:]) != expected_shape:
        raise ValueError('Dataset sample shape does not match its registered definition')
    args.input_shape, args.class_count = expected_shape, definition.class_count
    state_path = args.output / 'model_state_v3.pt'
    stored_manifest_path = args.output / 'workflow_manifest_v3.json'
    stored = json.loads(stored_manifest_path.read_text()) if stored_manifest_path.exists() else old or {}
    checkpoint_hash = stored.get('checkpoint_sha256')
    weights_source = stored.get('weights_source', 'checkpoint' if checkpoint_hash else 'seeded_random')
    if args.checkpoint is not None:
        if not args.checkpoint.is_file():
            raise ValueError('Checkpoint does not exist')
        checkpoint_hash = digest(args.checkpoint)
        weights_source = 'checkpoint'
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    elif state_path.exists():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
    elif old and old.get('checkpoint_sha256'):
        raise ValueError('Supply --checkpoint to reuse this historical decomposition; no saved model state exists yet')
    else:
        state = None
    model = build_model(args.model, definition, private, args.seed, state)
    manifest = dict(protocol='coalition_workflow_v3', model=args.model, dataset=args.dataset,
        weights_source=weights_source, checkpoint_sha256=checkpoint_hash,
        model_state_sha256=model_digest(model), seed=args.seed, private_index=private_index,
        private_label=int(y), public_indices=public_indices, public_labels=public_labels,
        arguments={name: getattr(args, name) for name in GENERATION_DEFAULTS},
        input_shape=list(expected_shape), class_count=definition.class_count,
        private_split='test', public_split='train', preprocessing='ToTensor raw [0,1]', loss='cross_entropy',
        private_tensor_sha256=tensor_digest(private), public_tensor_sha256=tensor_digest(public),
        parameter_order=[dict(name=k, shape=list(v.shape)) for k, v in model.named_parameters()])
    if old and old.get('protocol') == 'sparse_disguise_v1':
        if args.model != 'lenet5' or args.dataset != 'cifar10':
            raise ValueError('Historical decomposition is LeNet5/CIFAR10; use a new --output for another model/dataset')
        wanted = dict(private_index=private_index, **manifest['arguments'])
        if (old['arguments'] != wanted or old['checkpoint_sha256'] != checkpoint_hash
                or old['public_indices'] != public_indices or old['public_labels'] != public_labels
                or old['parameter_order'] != manifest['parameter_order']):
            raise ValueError('Historical decomposition inputs differ; choose a fresh --output')
        files = args.data / 'cifar-10-batches-py'
        current_hashes = {p.name: digest(p) for p in files.iterdir() if p.is_file()}
        if current_hashes != old['dataset_files_sha256']:
            raise ValueError('Historical CIFAR data hash mismatch')
    elif old and old != manifest:
        raise ValueError('Model/dataset/sample/decomposition settings differ; choose a fresh --output')
    if stored_manifest_path.exists() and stored != manifest:
        raise ValueError('Saved workflow inputs differ; choose a fresh --output')
    if not old:
        if any(args.output.iterdir()):
            raise ValueError('Refusing a nonempty output without a manifest')
        write_json(args.output / 'manifest.json', manifest)
    if not stored_manifest_path.exists():
        write_json(stored_manifest_path, manifest)
    if state_path.exists():
        # Validate even when an external checkpoint was supplied on resume.
        saved = torch.load(state_path, map_location='cpu', weights_only=True)
        if any(k not in saved or not torch.equal(v.cpu(), saved[k].cpu()) for k, v in model.state_dict().items()) or set(saved) != set(model.state_dict()):
            raise ValueError('Saved model state differs; refusing resume')
    else:
        torch.save(model.state_dict(), state_path)
    print(f'Model={args.model}; dataset={args.dataset}; seed={args.seed}; private test index={private_index}; weights={weights_source}', flush=True)
    return model, private, public, label, public_indices, public_labels, manifest
