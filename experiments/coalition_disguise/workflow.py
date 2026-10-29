"""The experiment's sequential workflow; fixed gradient-nearest disguise per subset."""
import argparse
import csv
import json
import math

import torch
from pathlib import Path
from torchvision.utils import save_image

from data.dataset_registry import dataset_dict
from models.model_registry import model_dict
from .attacks import attack, build_attack, config_hash, execution_config
from .common import digest, gradient, tensor_digest, write_json
from .diagnostics import input_checks, inspect_saved_runs
from .geometry import observe, solve
from .inputs import prepare_inputs, resolve_settings
from .residual_solver import solve_residual


def subset_plan(artifact, checks, public_indices, public_labels):
    """Freeze argmin gradient distances before any reconstruction is run."""
    plans = []
    for row in checks['coalition_geometry']:
        slot = row['public_slot']
        observation = observe(artifact['shares'], row['servers'])
        plan = dict(subset=f"{row['coalition']:0{len(artifact['shares'])}b}",
            servers=row['servers'], disguise_slot=slot,
            disguise_index=public_indices[slot], disguise_label=public_labels[slot],
            gradient_distance_to_disguise=row['public_distance'],
            gradient_distance_to_ground_truth=row['private_distance'],
            gradient_margin=row['margin'], observation_sha256=tensor_digest(observation))
        if 'decomposition_mode' in artifact:
            plan.update(gradient_l2_to_disguise=row['public_norm'], gradient_l2_to_ground_truth=row['private_norm'],
                disguise_residual_nnz=row['disguise_residual_nnz'],
                ground_truth_residual_nnz=row['ground_truth_residual_nnz'],
                residual_nnz_advantage=row['residual_nnz_advantage'],
                residual_sparsity_required=artifact.get('residual_nnz_gap') is not None,
                residual_sparsity_pass=row['residual_sparsity_pass'])
        plans.append(plan)
    return plans


def evaluate_result(model, image, private, public, label, observation, artifact, public_indices, best_metrics, assigned=None):
    from .run import evaluate
    # Limit evaluation to the preassigned image. No image-space reselection.
    selected = public if assigned is None else public[assigned['disguise_slot']:assigned['disguise_slot']+1]
    indices = public_indices if assigned is None else [assigned['disguise_index']]
    raw = evaluate(model, image, private, selected, label, observation, artifact['g'], indices, best_metrics)
    result = {key: raw[key] for key in ('private_mse', 'private_ssim', 'clamped_private_mse',
        'gradient_l2_squared', 'true_label_gradient_l2_squared', 'oracle_private_gradient_l2_squared',
        'pixel_min', 'pixel_max', 'outside_pixel_fraction')}
    result.update(ground_truth_mse=raw['private_mse'], ground_truth_ssim=raw['private_ssim'])
    if assigned is not None:
        result.update(disguise_mse=raw['public_min_mse'], disguise_ssim=raw['public_max_ssim'],
            disguise_mse_advantage=raw['private_mse']-raw['public_min_mse'],
            disguise_ssim_advantage=raw['public_max_ssim']-raw['private_ssim'], **assigned)
    return result


def save_result(destination, summary, image, model, private, public, label, observation, artifact, public_indices, args, manifest, assigned=None):
    summary.update(workflow_protocol='fixed_disguise_v3', model=args.model, dataset=args.dataset,
                   seed=args.seed, private_index=manifest['private_index'], decomposition=args.decomposition,
                   residual_nnz_gap=args.residual_nnz_gap if args.decomposition == 'residual' else None,
                   max_disguise_distance_squared=args.max_disguise_distance_squared)
    summary['evaluation'] = evaluate_result(model, image, private, public, label, observation,
                                            artifact, public_indices, summary['best_metrics'], assigned)
    write_json(destination / 'summary.json', summary)
    images = [private, image]
    if assigned is not None:
        images.append(public[assigned['disguise_slot']][None])
    save_image(torch.cat(images).clamp(0, 1), destination / 'comparison.png', nrow=len(images))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=sorted(model_dict), help='Default: lenet5, or the saved run model')
    parser.add_argument('--dataset', choices=sorted(dataset_dict), help='Default: cifar10, or the saved run dataset')
    parser.add_argument('--checkpoint', type=Path, help='Optional state_dict; otherwise use seeded random weights or the saved run model')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--download', action='store_true', help='Allow torchvision to download a missing dataset')
    parser.add_argument('--seed', type=int, help='Default: 2026; controls independent model, sample and public-reference streams')
    parser.add_argument('--private-index', type=int, help='Optional override; otherwise draw a seeded random test sample')
    parser.add_argument('--public-count', type=int, help='Number of distinct public disguises (default 100)')
    parser.add_argument('--servers', type=int, choices=range(2, 9), help='Default 3; enumerate every nonempty proper subset')
    parser.add_argument('--relative-margin', type=float)
    parser.add_argument('--partitions', type=int)
    parser.add_argument('--anchors', type=int)
    parser.add_argument('--decomposition', choices=['residual', 'sparse'],
                        help='Default: residual (shorter and sparser differences); saved runs retain their construction')
    parser.add_argument('--residual-nnz-gap', type=int,
                        help='Require at least this many fewer nonzeros toward the nearest disguise (default 1; 0 allows ties)')
    parser.add_argument('--max-disguise-distance-squared', type=float,
                        help='Optional hard cap on every subset\'s squared distance to its nearest disguise gradient')
    parser.add_argument('--presets', nargs='+', choices=['idlg', 'dlg', 'geiping'], default=['idlg'])
    parser.add_argument('--attack-profile', choices=['calibrated', 'preset'], default='calibrated')
    parser.add_argument('--initialization', choices=['random', 'uniform', 'uniform_sigmoid',
                        'smooth_uniform', 'smooth_uniform_sigmoid'],
                        help='Optional override; calibrated iDLG/DLG starts from a smooth seeded image')
    parser.add_argument('--restarts', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--minimum-change', type=float, default=1e-10)
    parser.add_argument('--lbfgs-tolerance-grad', type=float, default=1e-12)
    parser.add_argument('--lbfgs-tolerance-change', type=float, default=1e-15)
    parser.add_argument('--diagnostics-interval', type=int, default=20)
    parser.add_argument('--max-iterations', type=int, help='Optional per-restart safety budget; image MSE never stops optimization')
    parser.add_argument('--baseline-mse-target', type=float, default=.01,
                        help='Proceed to subsets only if full-share reconstruction MSE is below this value')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--diagnose-only', action='store_true')
    modes.add_argument('--baseline-only', action='store_true')
    args = parser.parse_args(argv)
    args.residual_gap_explicit = args.residual_nnz_gap is not None
    if not math.isfinite(args.baseline_mse_target) or args.baseline_mse_target <= 0:
        parser.error('baseline-mse-target must be finite and positive')
    return args


def execute(args):
    torch.set_num_threads(2)
    old = resolve_settings(args)
    if args.decomposition == 'sparse' and args.residual_gap_explicit:
        raise ValueError('--residual-nnz-gap requires the residual decomposition; it cannot be applied to legacy sparse shares')
    if (args.public_count < 1 or args.partitions < 1 or args.anchors < 1
            or not math.isfinite(args.relative_margin) or args.relative_margin < 0):
        raise ValueError('Public count/search budgets must be positive; relative margin must be finite and nonnegative')
    if args.residual_nnz_gap < 0:
        raise ValueError('Residual nonzero gap must be nonnegative')
    if args.max_disguise_distance_squared is not None and (
            not math.isfinite(args.max_disguise_distance_squared) or args.max_disguise_distance_squared < 0):
        raise ValueError('Maximum squared disguise distance must be finite and nonnegative')
    definition = dataset_dict[args.dataset]
    args.class_count = definition.class_count
    args.input_shape = (definition.input_channels, *definition.input_dimensions)
    if not args.prepare_only and not args.diagnose_only:
        for name in args.presets:
            build_attack(name, args)
    if args.diagnose_only and not (args.output / 'decomposition.pt').is_file():
        raise ValueError('No saved decomposition; --diagnose-only never creates shares')
    args.output.mkdir(parents=True, exist_ok=True)
    model, private, public, label, public_indices, public_labels, manifest = prepare_inputs(args, old)
    artifact_path = args.output / 'decomposition.pt'
    if artifact_path.exists():
        expected = json.loads((args.output / 'decomposition_hash.json').read_text())['sha256']
        if digest(artifact_path) != expected:
            raise ValueError('Saved decomposition hash mismatch')
        artifact = torch.load(artifact_path, map_location='cpu', weights_only=True)
        if (args.decomposition == 'residual' and artifact.get('residual_nnz_gap') != args.residual_nnz_gap
                or artifact.get('max_disguise_distance_squared') != args.max_disguise_distance_squared):
            raise ValueError('Saved decomposition constraints differ; choose a fresh --output')
        print(f'Reusing shares: {artifact_path}', flush=True)
    else:
        g = gradient(model, private, label)
        d = torch.stack([gradient(model, image[None], torch.tensor([target]))
                         for image, target in zip(public, public_labels)])
        try:
            options = dict(max_disguise_distance_squared=args.max_disguise_distance_squared)
            solver = solve
            if args.decomposition == 'residual':
                solver = solve_residual
                options['residual_nnz_gap'] = args.residual_nnz_gap
            artifact = solver(g, d, args.servers, args.seed, args.relative_margin, args.partitions, args.anchors, **options)
        except RuntimeError as error:
            failure = dict(state='decomposition_failed', message=str(error),
                           diagnostics=getattr(error, 'diagnostics', None))
            write_json(args.output / 'decomposition_failure.json', failure)
            print(f'Decomposition failed; no reconstruction jobs launched. See {args.output / "decomposition_failure.json"}', flush=True)
            raise
        artifact.update(g=g, d=d, decomposition_mode=args.decomposition,
                        max_disguise_distance_squared=args.max_disguise_distance_squared)
        torch.save(artifact, artifact_path)
        write_json(args.output / 'decomposition_hash.json', dict(sha256=digest(artifact_path)))
        (args.output / 'decomposition_failure.json').unlink(missing_ok=True)
    if 'search_diagnostics' in artifact:
        write_json(args.output / 'decomposition_search.json', artifact['search_diagnostics'])
    checks = input_checks(model, private, label, artifact)
    write_json(args.output / 'input_checks_v3.json', checks)
    if args.diagnose_only:
        result = dict(input_checks=checks, saved_runs=inspect_saved_runs(args.output, model, private, label, artifact))
        write_json(args.output / 'diagnosis_v3.json', result)
        print(json.dumps(result, indent=2))
        return result
    audit = checks['decomposition']
    if not (checks['checkpoint_data_gradient_equal'] and checks['flatten_unpack_roundtrip_equal']
            and audit['all_coalitions_pass'] and audit['float32_aggregation_pass'] and audit['float64_aggregation_pass']):
        raise RuntimeError('Decomposition/input verification failed; see input_checks_v3.json')
    plans = subset_plan(artifact, checks, public_indices, public_labels)
    plan_path = args.output / 'subsets_v3.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plans:
        raise ValueError('Saved subset/disguise assignment differs; refusing to replace it')
    if not plan_path.exists():
        write_json(plan_path, plans)
    save_image(private, args.output / 'ground_truth.png')
    save_image(public, args.output / 'disguises.png')
    print(f'Verified decomposition; {len(plans)} proper subsets, each with a fixed gradient-nearest disguise.', flush=True)
    if audit['residual_sparsity_required']:
        print(f'Residual checks passed for every subset: minimum nonzero advantage={audit["minimum_residual_nnz_advantage"]}; '
              f'worst squared disguise distance={audit["maximum_disguise_distance_squared"]:.6g}.', flush=True)
    if args.prepare_only:
        return checks
    from .run import refresh_report
    full = artifact['shares'].sum(0)
    statuses = []
    for name in args.presets:
        executable = build_attack(name, args)
        config = execution_config(executable, args)
        folder = args.output / f'{name}_{config_hash(config)}'
        folder.mkdir(exist_ok=True)
        print(f'{name}: reconstructing full-share ground-truth control', flush=True)
        destination = folder / 'full'
        summary, image = attack(model, full, label, name, args, destination)
        summary = save_result(destination, summary, image, model, private, public, label, full,
                              artifact, public_indices, args, manifest)
        mse = summary['evaluation']['ground_truth_mse']
        passed = mse < args.baseline_mse_target
        status = dict(preset=name, model=args.model, dataset=args.dataset, seed=args.seed,
                      private_index=manifest['private_index'], baseline_mse=mse,
                      baseline_mse_target=args.baseline_mse_target, baseline_passed=passed,
                      subset_count=len(plans), completed_subsets=0,
                      state='baseline_passed' if passed else 'baseline_failed')
        write_json(folder / 'baseline.json', status)
        refresh_report(args.output)
        print(f'{name}: baseline MSE={mse:.6g}; target < {args.baseline_mse_target}; passed={passed}', flush=True)
        if passed and not args.baseline_only:
            for assigned in plans:
                print(f'{name}: subset {assigned["subset"]}, fixed disguise train index {assigned["disguise_index"]}', flush=True)
                observation = observe(artifact['shares'], assigned['servers'])
                destination = folder / assigned['subset']
                summary, image = attack(model, observation, label, name, args, destination)
                save_result(destination, summary, image, model, private, public, label, observation,
                            artifact, public_indices, args, manifest, assigned)
                status['completed_subsets'] += 1
                write_json(folder / 'baseline.json', status)
                refresh_report(args.output)
            status['state'] = 'complete'
            write_json(folder / 'baseline.json', status)
        elif not passed:
            print('Baseline failed; no subset attacks launched.', flush=True)
        statuses.append(status)
    write_json(args.output / 'workflow_status_v3.json', statuses)
    print(f'Report: {(args.output / "report.csv").resolve()}', flush=True)
    return statuses


def main(argv=None):
    return execute(parse_args(argv))
