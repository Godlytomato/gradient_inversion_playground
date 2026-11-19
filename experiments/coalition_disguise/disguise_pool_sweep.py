"""Test a fixed pool of public disguises against 256 distinct private CIFAR-10 images."""
import argparse
import json
from pathlib import Path

import torch

from data.dataset_registry import dataset_dict
from .common import gradient, model_digest, tensor_digest
from .diagnostics import input_checks
from .inputs import build_model, load_datasets, select_indices
from .residual_solver import ResidualSearchError, solve_residual
from .workflow import subset_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--public-count', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed = 2026
    private, public = load_datasets('cifar10', args.data, args.download)
    rng = torch.Generator().manual_seed(256)
    targets = torch.randperm(len(private), generator=rng)[:256].tolist()
    _, indices = select_indices(len(private), len(public), args.public_count, seed, targets[0])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    model = build_model('lenet5', dataset_dict['cifar10'], private[targets[0]][0][None], seed, state)
    labels = [int(public[i][1]) for i in indices]
    references = torch.stack([gradient(model, public[i][0][None], torch.tensor([y]))
                              for i, y in zip(indices, labels)])
    manifest = dict(model=model_digest(model), public_indices=indices, private_indices=targets,
                    gradients=tensor_digest(references), seed=seed, servers=3,
                    relative_margin=.01, partitions=8, anchors=16, residual_nnz_gap=1)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Inputs changed. Choose a fresh output directory.')
    path.write_text(json.dumps(manifest, indent=2))
    passed = 0
    for count, index in enumerate(targets, 1):
        path = args.output / f'image_{index}.json'
        if path.exists():
            row = json.loads(path.read_text())
        else:
            image, y = private[index]
            image, label = image[None], torch.tensor([int(y)])
            g = gradient(model, image, label)
            row = dict(private_index=index, label=int(y))
            try:
                artifact = solve_residual(g, references, 3, seed, .01, 8, 16, residual_nnz_gap=1,
                                          max_disguise_distance_squared=None)
            except ResidualSearchError as error:
                row.update(status='search_failed', reason=str(error))
            else:
                artifact.update(g=g, d=references, decomposition_mode='residual',
                                max_disguise_distance_squared=None)
                checks = input_checks(model, image, label, artifact)
                audit = checks['decomposition']
                if not all([checks['checkpoint_data_gradient_equal'], checks['flatten_unpack_roundtrip_equal'],
                            audit['float32_aggregation_pass'], audit['float64_aggregation_pass']]):
                    raise RuntimeError(f'Input or aggregation verification failed for index {index}')
                subsets = subset_plan(artifact, checks, indices, labels)
                if len(subsets) != 6:
                    raise RuntimeError('Expected six proper subsets')
                row.update(status='pass' if audit['all_coalitions_pass'] else 'geometry_failed',
                           subsets=subsets)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(row, indent=2))
            temp.replace(path)
        passed += row['status'] == 'pass'
        print(f'{count}/256 images checked, {passed} passed', flush=True)


if __name__ == '__main__':
    main()
