"""Single-image gradient disguise experiment, paired controls, and diagnostics."""
import argparse
import csv
import json
import platform
from pathlib import Path

import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image

from .common import digest, gradient, pixel_metrics, unpack, write_json
from .attacks import attack, build_attack, config_hash, execution_config, TraceObserver
from .diagnostics import input_checks, inspect_saved_runs, calibration_diagnosis
from .geometry import solve, observe, coalitions
from models.lenet import LeNet5
from observer.metrics.implementations.ssim import ssim


def claim_manifest(directory, manifest):
    """Decomposition identity excludes execution code/version provenance.

    Historical manifests are read but never rewritten. Saved tensor hashes and
    current checkpoint/data gradients are independently checked before reuse.
    """
    path = directory / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        provenance = {'source_sha256', 'torch_version', 'python_version'}
        identity = lambda value: {k: v for k, v in value.items() if k not in provenance}
        if identity(old) != identity(manifest):
            raise ValueError('Decomposition inputs differ; choose a fresh output directory')
    else:
        if any(directory.iterdir()):
            raise ValueError('Refusing a nonempty output directory without a manifest')
        write_json(path, manifest)


def evaluate(model, image, private, public, label, observation, g, public_indices, best_metrics=None):
    private_mse = float((image - private).square().mean())
    private_ssim = ssim(image, private)
    mses = (public - image).square().flatten(1).mean(1)
    ssims = [ssim(image, y[None]) for y in public]
    mse_slot = int(mses.argmin())
    ssim_slot = max(range(len(ssims)), key=ssims.__getitem__)
    metrics = best_metrics or {}
    if 'label_probabilities' in metrics:
        attack_label = torch.tensor(metrics['label_probabilities'], dtype=image.dtype)
    elif 'selected_label' in metrics:
        attack_label = torch.tensor([metrics['selected_label']])
    else:
        attack_label = label
    recovered_gradient = gradient(model, image, attack_label)
    true_label_gradient = gradient(model, image, label)
    return dict(private_mse=private_mse, private_ssim=private_ssim,
                clamped_private_mse=float((image.clamp(0, 1) - private).square().mean()),
                public_min_mse=float(mses[mse_slot]), public_max_ssim=ssims[ssim_slot],
                public_mse_index=public_indices[mse_slot], public_ssim_index=public_indices[ssim_slot],
                public_mse_advantage=private_mse - float(mses[mse_slot]),
                public_ssim_advantage=ssims[ssim_slot] - private_ssim,
                gradient_l2_squared=float((recovered_gradient.double() - observation.double()).square().sum()),
                true_label_gradient_l2_squared=float((true_label_gradient.double() - observation.double()).square().sum()),
                oracle_private_gradient_l2_squared=float((g.double() - observation.double()).square().sum()),
                **pixel_metrics(image))


def refresh_report(root):
    """Include historical and new configurations; do not erase earlier results."""
    records = []
    for path in sorted(root.glob('*/*/summary.json')):
        summary = json.loads(path.read_text())
        if 'evaluation' not in summary:
            continue
        folder = path.parent.parent.name
        name, _, hashed = folder.partition('_')
        records.append(dict(preset=summary.get('preset', name),
            profile=summary.get('profile', 'historical_preset'),
            configuration=summary.get('configuration', hashed), coalition=path.parent.name,
            attack_loss=summary['best_metrics']['attack_loss'],
            iterations=summary['total_iterations'], stopping_reason=summary['reason'],
            restarts=summary['run_metrics']['restarts_started'], **summary['evaluation']))
    write_json(root / 'report.json', records)
    if records:
        fields = list(dict.fromkeys(key for record in records for key in record))
        with (root / 'report.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return records


def save_evaluation(destination, summary, image, model, private, public, label, observation, artifact, public_indices):
    summary['evaluation'] = evaluate(model, image, private, public, label, observation, artifact['g'], public_indices, summary['best_metrics'])
    write_json(destination / 'summary.json', summary)
    match = public_indices.index(summary['evaluation']['public_ssim_index'])
    save_image(torch.cat([private, image, public[match][None]]).clamp(0, 1),
               destination / 'comparison.png', nrow=3)
    return summary


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
    parser.add_argument('--attack-profile', choices=['calibrated', 'preset'], default='calibrated',
                        help='calibrated bounds the image search and tightens LBFGS tolerances; preset preserves registered settings')
    parser.add_argument('--restarts', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--minimum-change', type=float, default=1e-10)
    parser.add_argument('--lbfgs-tolerance-grad', type=float, default=1e-12)
    parser.add_argument('--lbfgs-tolerance-change', type=float, default=1e-15)
    parser.add_argument('--diagnostics-interval', type=int, default=20)
    parser.add_argument('--max-iterations', type=int, help='Optional safety cap; exhausting it fails calibration')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--diagnose-only', action='store_true', help='Inspect existing artifacts; never rerun an attack or regenerate shares')
    parser.add_argument('--baseline-only', action='store_true', help='Run paired full/original controls and stop before coalitions')
    args = parser.parse_args()
    if sum((args.prepare_only, args.diagnose_only, args.baseline_only)) > 1:
        parser.error('Choose only one of --prepare-only, --diagnose-only, --baseline-only')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if not args.checkpoint.is_file():
        parser.error('Checkpoint is missing. Supply trained weights; random weights are not a calibrated baseline.')
    if not args.diagnose_only and not args.prepare_only:
        for name in args.presets:
            build_attack(name, args)  # Validate settings before any expensive work.
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
    model(private)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    decomposition_args = {'private_index', 'public_count', 'seed', 'servers', 'relative_margin', 'partitions', 'anchors'}
    manifest = dict(protocol='sparse_disguise_v1', checkpoint_sha256=digest(args.checkpoint),
                    arguments={k: v for k, v in vars(args).items() if k in decomposition_args},
                    public_indices=public_indices, public_labels=public_labels,
                    parameter_order=[dict(name=k, shape=list(v.shape)) for k, v in model.named_parameters()],
                    preprocessing='torchvision.ToTensor raw [0,1]', loss='cross_entropy',
                    torch_version=str(torch.__version__), python_version=platform.python_version(),
                    source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                    dataset_files_sha256={p.name: digest(p) for p in (args.data / 'cifar-10-batches-py').iterdir() if p.is_file()})
    if args.diagnose_only and not (args.output / 'decomposition.pt').is_file():
        parser.error('No saved decomposition in --output; --diagnose-only never creates one')
    args.output.mkdir(parents=True, exist_ok=True)
    claim_manifest(args.output, manifest)
    artifact_path = args.output / 'decomposition.pt'
    if artifact_path.exists():
        if digest(artifact_path) != json.loads((args.output / 'decomposition_hash.json').read_text())['sha256']:
            raise ValueError('Decomposition artifact changed')
        artifact = torch.load(artifact_path, weights_only=True)
        print(f'Reusing decomposition: {artifact_path}', flush=True)
    else:
        print('Computing private/public gradients and searching for shares...', flush=True)
        g = gradient(model, private, label)
        d = torch.stack([gradient(model, im[None], torch.tensor([lab])) for im, lab in zip(public, public_labels)])
        artifact = solve(g, d, args.servers, args.seed, args.relative_margin, args.partitions, args.anchors)
        artifact.update(g=g, d=d)
        torch.save(artifact, artifact_path)
        write_json(args.output / 'decomposition_hash.json', dict(sha256=digest(artifact_path)))
    g, d, shares = (artifact[k] for k in ('g', 'd', 'shares'))
    checks = input_checks(model, private, label, artifact)
    report = checks['decomposition']
    if args.diagnose_only:
        result = dict(input_checks=checks, saved_runs=inspect_saved_runs(args.output, model, private, label, artifact))
        path = args.output / 'diagnosis_v2.json'
        write_json(path, result)
        print(json.dumps(result, indent=2))
        print(f'Diagnostics: {path}')
        return
    write_json(args.output / 'input_checks_v2.json', checks)
    if not checks['checkpoint_data_gradient_equal'] or not checks['flatten_unpack_roundtrip_equal']:
        raise ValueError('Saved gradient does not match current checkpoint/data or parameter ordering; see input_checks_v2.json')
    if not report['all_coalitions_pass'] or not report['float32_aggregation_pass'] or not report['float64_aggregation_pass']:
        raise RuntimeError('Serialized decomposition failed verification; see input_checks_v2.json')
    # Version new audits; never rewrite the v1 audit or immutable shares.
    write_json(args.output / 'audit_v2.json', report)
    geometry_path = args.output / 'coalition_geometry_v2.csv'
    with geometry_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks['coalition_geometry'][0]))
        writer.writeheader()
        writer.writerows(checks['coalition_geometry'])
    if args.prepare_only:
        print(json.dumps(report, indent=2))
        return
    full = shares.sum(0)
    for name in args.presets:
        executable = build_attack(name, args)
        config = execution_config(executable, args)
        folder = args.output / f'{name}_{config_hash(config)}'
        folder.mkdir(exist_ok=True)
        summaries, images = {}, {}
        # The saved-share positive control always runs first. The original G
        # is then an explicitly identified undecomposed diagnostic control.
        for control, observation in [('full', full), ('original_control', g)]:
            print(f'{name} ({args.attack_profile}): {control}', flush=True)
            destination = folder / control
            summary, image = attack(model, observation, label, name, args, destination)
            summary['observation_role'] = 'saved_share_sum' if control == 'full' else 'undecomposed_gradient_diagnostic_control'
            summaries[control] = save_evaluation(destination, summary, image, model, private, public, label, observation, artifact, public_indices)
            images[control] = image
            refresh_report(args.output)
        decision = calibration_diagnosis(checks, summaries['full'], summaries['original_control'], executable.description.attack.restarts)
        decision['full_vs_original_reconstruction_mse'] = float((images['full'] - images['original_control']).square().mean())
        write_json(folder / 'calibration.json', decision)
        print(f'{name}: stage={decision["stage"]}; full MSE={decision["full_private_mse"]:.6g}; original MSE={decision["original_private_mse"]:.6g}', flush=True)
        if not decision['calibrated'] or args.baseline_only:
            print('Coalition attacks skipped; see calibration.json and trajectory.csv.', flush=True)
            continue
        for subset in coalitions(len(shares)):
            coalition = f'{sum(1 << i for i in subset):03b}'
            print(f'{name}: coalition {coalition}', flush=True)
            observation = observe(shares, subset)
            destination = folder / coalition
            summary, image = attack(model, observation, label, name, args, destination)
            save_evaluation(destination, summary, image, model, private, public, label, observation, artifact, public_indices)
            refresh_report(args.output)
    print(f'Report: {(args.output / "report.json").resolve()}')


if __name__ == '__main__':
    main()
