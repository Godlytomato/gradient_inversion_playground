"""Run from repository root: python -m experiments.coalition_disguise.run --help."""
import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from .geometry import solve, audit, observe, coalitions
from models.lenet import LeNet5
from observer.observer import observer
from observer.metrics.implementations.ssim import ssim
from run_def import (run_desc, batch, observer_description, function_package,
                     best_selection_description)
from run_utils.run_builder import build_run
from validation.registries.registry_builder import build_registries


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def claim_manifest(directory, manifest):
    """Refuse resume under different inputs/configuration; never erase a run."""
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Output manifest differs; choose a fresh output directory')
    else:
        if any(directory.iterdir()):
            raise ValueError('Refusing a nonempty output directory without a manifest')
        write_json(path, manifest)


def gradient(model, image, label):
    loss = F.cross_entropy(model(image), label)
    return torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])


def unpack(vector, model):
    result, offset = [], 0
    for p in model.parameters():
        result.append(vector[offset:offset+p.numel()].reshape_as(p).clone())
        offset += p.numel()
    if offset != len(vector):
        raise ValueError('Gradient parameter ordering/size mismatch')
    return result


def build_attack(name, args):
    preset = build_registries()['attacks'][name]
    job = preset.builder()
    job.base_seed = args.seed
    if args.restarts is not None:
        job.restarts = args.restarts
    if args.lr is not None:
        job.optimizer_description.parameters['lr'] = args.lr
    config = observer_description(
        fail_policies=[function_package('no_improvement', dict(metric='attack_loss',
                            patience=args.patience, minimum_change=args.minimum_change)),
                       function_package('max_iterations', dict(limit=args.max_iterations or preset.max_iterations))],
        best_selection=best_selection_description('attack_loss'))
    description = run_desc(name='coalition_disguise_' + name, model='lenet5',
                           data=batch('cifar10', 1, list(range(10))),
                           gradient_transformations=[], attack=job, attack_name=name,
                           observer=config, experiment_seed=args.seed)
    executable = build_run(description)
    # Local compatibility adapter: validate all components first, then explicitly
    # replace only the experiment's resolved/recorded safety policy. No shared edits.
    if args.max_iterations is None:
        executable.observer.fail_policies = [p for p in executable.observer.fail_policies if p.name != 'max_iterations']
        description.observer.fail_policies = [p for p in description.observer.fail_policies if p.name != 'max_iterations']
    if executable.requirements.attack_information - {
            'model', 'gradients', 'batch_size', 'input_shape', 'class_count', 'device', 'labels'}:
        raise ValueError('Preset requests information outside the attacker allowlist')
    return executable


class TraceObserver(observer):
    """Use existing selection/stopping logic; add a local objective trajectory."""
    def __init__(self, config, path):
        super().__init__(config=config)
        self.stream = path.open('w')
        self.stream.write('step,restart,attack_loss\n')
        self.attempt_reasons = []

    def update(self, reconstruction, metrics=None):
        super().update(reconstruction, metrics)
        self.stream.write(f"{self.step},{metrics['restart_index']},{metrics['attack_loss']}\n")
        if self.should_stop():
            self.attempt_reasons.append(self.outcome_reason)

    def close(self):
        self.stream.close()
        super().close()


def attack(model, observation, label, name, args, destination):
    destination.mkdir(exist_ok=True)
    summary_path = destination / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if digest(destination / 'reconstruction.pt') != summary['reconstruction_sha256']:
            raise ValueError('Saved reconstruction changed; refusing resume')
        return summary, torch.load(destination / 'reconstruction.pt', weights_only=True)
    executable = build_attack(name, args)
    write_json(destination / 'attack_config.json', asdict(executable.description))
    from transformations.transform_def import grad_package
    information = dict(model=model, gradients=grad_package(unpack(observation, model)),
                       batch_size=1, input_shape=(3, 32, 32), class_count=10,
                       device=torch.device('cpu'), labels=label)
    # Original image, original gradient and public images are absent here.
    information = {key: information[key] for key in executable.requirements.attack_information}
    active = TraceObserver(executable.observer, destination / 'trajectory.csv')
    try:
        image = executable.functions.attack.function(information=information,
                    components=executable.functions.components, observer=active,
                    **executable.functions.attack.parameters)
        summary = active.summary()
        summary['attempt_stopping_reasons'] = active.attempt_reasons
    finally:
        active.close()
    torch.save(image, destination / 'reconstruction.pt')
    save_image(image.clamp(0, 1), destination / 'reconstruction.png')
    summary['reconstruction_sha256'] = digest(destination / 'reconstruction.pt')
    return summary, image


def evaluate(model, image, private, public, label, observation, g, public_indices):
    private_mse = float((image - private).square().mean())
    private_ssim = ssim(image, private)
    mses = (public - image).square().flatten(1).mean(1)
    ssims = [ssim(image, y[None]) for y in public]
    mse_slot = int(mses.argmin())
    ssim_slot = max(range(len(ssims)), key=ssims.__getitem__)
    recovered_gradient = gradient(model, image, label)
    return dict(private_mse=private_mse, private_ssim=private_ssim,
                public_min_mse=float(mses[mse_slot]), public_max_ssim=ssims[ssim_slot],
                public_mse_index=public_indices[mse_slot], public_ssim_index=public_indices[ssim_slot],
                public_mse_advantage=private_mse - float(mses[mse_slot]),
                public_ssim_advantage=ssims[ssim_slot] - private_ssim,
                gradient_l2_squared=float((recovered_gradient.double() - observation.double()).square().sum()),
                oracle_private_gradient_l2_squared=float((g.double() - observation.double()).square().sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Trained CIFAR RGB LeNet5 state_dict')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--private-index', type=int, default=193)
    parser.add_argument('--public-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--servers', type=int, default=3, choices=[3])
    parser.add_argument('--relative-margin', type=float, default=.01)
    parser.add_argument('--partitions', type=int, default=8)
    parser.add_argument('--anchors', type=int, default=16)
    parser.add_argument('--presets', nargs='+', choices=['idlg', 'dlg', 'geiping'], default=['idlg'])
    parser.add_argument('--restarts', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--minimum-change', type=float, default=1e-7)
    parser.add_argument('--max-iterations', type=int, help='Optional smoke/safety cap; capped runs cannot pass calibration')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if not args.checkpoint.is_file():
        parser.error('Checkpoint is missing. Supply trained weights; random weights are not a calibrated baseline.')
    # Require local data, never implicitly download or alter existing datasets.
    train = CIFAR10(args.data, train=True, download=False, transform=ToTensor())
    test = CIFAR10(args.data, train=False, download=False, transform=ToTensor())
    if not 1 <= args.public_count <= len(train) or not 0 <= args.private_index < len(test):
        parser.error('Dataset indices/count are out of range')
    private, y = test[args.private_index]
    private, label = private[None], torch.tensor([y])
    rng = torch.Generator().manual_seed(args.seed)
    public_indices = torch.randperm(len(train), generator=rng)[:args.public_count].tolist()
    public = torch.stack([train[i][0] for i in public_indices])
    public_labels = [train[i][1] for i in public_indices]
    model = LeNet5()
    model.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
    model(private)  # materialize LazyLinear before strict checkpoint validation
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    manifest = dict(protocol='sparse_disguise_v1', checkpoint_sha256=digest(args.checkpoint),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                               if k not in {'output', 'prepare_only', 'checkpoint', 'data', 'presets', 'restarts', 'lr',
                                            'patience', 'minimum_change', 'max_iterations'}},
                    public_indices=public_indices, public_labels=public_labels,
                    parameter_order=[dict(name=k, shape=list(v.shape)) for k, v in model.named_parameters()],
                    preprocessing='torchvision.ToTensor raw [0,1]', loss='cross_entropy',
                    torch_version=str(torch.__version__), python_version=platform.python_version(),
                    source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                    dataset_files_sha256={p.name: digest(p) for p in (args.data / 'cifar-10-batches-py').iterdir() if p.is_file()})
    args.output.mkdir(parents=True, exist_ok=True)
    claim_manifest(args.output, manifest)
    artifact_path = args.output / 'decomposition.pt'
    if artifact_path.exists():
        if digest(artifact_path) != json.loads((args.output / 'decomposition_hash.json').read_text())['sha256']:
            raise ValueError('Decomposition artifact changed')
        artifact = torch.load(artifact_path, weights_only=True)
    else:
        g = gradient(model, private, label)
        d = torch.stack([gradient(model, im[None], torch.tensor([lab])) for im, lab in zip(public, public_labels)])
        artifact = solve(g, d, args.servers, args.seed, args.relative_margin, args.partitions, args.anchors)
        artifact.update(g=g, d=d)
        torch.save(artifact, artifact_path)
        write_json(args.output / 'decomposition_hash.json', dict(sha256=digest(artifact_path)))
    g, d, shares = (artifact[k] for k in ('g', 'd', 'shares'))
    report, rows = audit(shares, g, d, artifact['epsilon'], artifact['exact_shares'])
    if not torch.equal(g, gradient(model, private, label)):
        raise ValueError('Saved private gradient does not match current model/data')
    write_json(args.output / 'audit.json', report)
    with (args.output / 'coalition_geometry.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not report['all_coalitions_pass'] or not report['float32_aggregation_pass']:
        raise RuntimeError('Serialized decomposition failed verification')
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    all_results = []
    full = shares.sum(0)  # Positive control uses actual serialized shares.
    for name in args.presets:
        executable = build_attack(name, args)
        config = asdict(executable.description)
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        folder = args.output / f'{name}_{config_hash}'
        folder.mkdir(exist_ok=True)
        observations = [('full', full)] + [(f'{sum(1 << i for i in s):03b}', observe(shares, s)) for s in coalitions(len(shares))]
        for coalition, observation in observations:
            destination = folder / coalition
            summary, image = attack(model, observation, label, name, args, destination)
            summary['evaluation'] = evaluate(model, image, private, public, label, observation, g, public_indices)
            write_json(destination / 'summary.json', summary)
            save_image(torch.cat([private, image, public[public_indices.index(summary['evaluation']['public_ssim_index'])][None]]).clamp(0, 1),
                       destination / 'comparison.png', nrow=3)
            record = dict(preset=name, configuration=config_hash, coalition=coalition,
                          attack_loss=summary['best_metrics']['attack_loss'],
                          iterations=summary['total_iterations'], stopping_reason=summary['reason'],
                          restarts=summary['run_metrics']['restarts_started'], **summary['evaluation'])
            all_results.append(record)
            write_json(args.output / 'report.json', all_results)
            with (args.output / 'report.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
                writer.writerows(all_results)
            if coalition == 'full':
                calibrated = (summary['evaluation']['private_mse'] < .01 and
                              all(r == 'no_improvement' for r in summary['attempt_stopping_reasons']))
                write_json(folder / 'calibration.json', dict(calibrated=calibrated, target_mse=.01))
                if not calibrated:
                    print(f'{name}: full-share baseline uncalibrated; skipping coalition attacks. See {destination}')
                    break
    print(f'Report: {args.output / "report.json"}')


if __name__ == '__main__':
    main()
