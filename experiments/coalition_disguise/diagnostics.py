"""Evaluation-only diagnostics. These functions never participate in optimization."""
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import gradient, pixel_metrics, unpack
from .geometry import audit


def input_checks(model, private, label, artifact):
    g, d, shares = (artifact[k] for k in ('g', 'd', 'shares'))
    full = shares.sum(0)
    recomputed = gradient(model, private, label)
    report, rows = audit(shares, g, d, artifact['epsilon'], artifact['exact_shares'],
                         residual_nnz_gap=artifact.get('residual_nnz_gap'),
                         max_disguise_distance_squared=artifact.get('max_disguise_distance_squared'))
    roundtrip = torch.cat([t.flatten() for t in unpack(full, model)])
    inferred = int(unpack(full, model)[-1].argmin())
    return dict(decomposition=report, coalition_geometry=rows,
        checkpoint_data_gradient_equal=bool(torch.equal(g, recomputed)),
        flatten_unpack_roundtrip_equal=bool(torch.equal(roundtrip, full)),
        true_label=int(label.item()), full_idlg_inferred_label=inferred,
        full_idlg_label_correct=inferred == int(label.item()),
        oracle_original_gradient_l2_squared=float((recomputed.double() - g.double()).square().sum()),
        oracle_full_gradient_l2_squared=float((recomputed.double() - full.double()).square().sum()))


def inspect_saved_runs(root, model, private, label, artifact):
    """Inspect old and new artifacts without modifying them or rerunning attacks."""
    reports = []
    for path in sorted(root.glob('*/*/summary.json')):
        image_path = path.parent / 'reconstruction.pt'
        if not image_path.exists():
            continue
        summary = json.loads(path.read_text())
        image = torch.load(image_path, map_location='cpu', weights_only=True)
        name = path.parent.name
        target = artifact['g'] if name == 'original_control' else artifact['shares'].sum(0) if name == 'full' else None
        data = dict(path=str(path.relative_to(root)), **pixel_metrics(image),
                    private_mse=float((image - private).square().mean()),
                    clamped_private_mse=float((image.clamp(0, 1) - private).square().mean()),
                    attack_loss=summary.get('best_metrics', {}).get('attack_loss'),
                    stopping_reasons=summary.get('attempt_stopping_reasons'))
        trajectory = path.parent / 'trajectory.csv'
        if trajectory.exists():
            with trajectory.open() as stream:
                trace = list(csv.DictReader(stream))
            values = [float(row['attack_loss']) for row in trace]
            trailing = 0
            for value in reversed(values):
                if value != values[-1]:
                    break
                trailing += 1
            data['identical_final_objective_rows'] = trailing
        if target is not None:
            # True label is used only here as an explicitly named diagnostic.
            z = image.detach().requires_grad_(True)
            grads = torch.autograd.grad(F.cross_entropy(model(z), label), tuple(model.parameters()), create_graph=True)
            loss = sum((a - b).square().sum() for a, b in zip(grads, unpack(target, model)))
            image_grad, = torch.autograd.grad(loss, z)
            data['true_label_l2_image_gradient_linf'] = float(image_grad.abs().max())
            data['true_label_gradient_l2_squared'] = float(loss.detach())
        reports.append(data)
    return reports


def calibrated(summary, expected_restarts):
    reasons = summary.get('attempt_stopping_reasons', [])
    return (summary['evaluation']['private_mse'] < .01 and
            summary['evaluation'].get('outside_pixel_fraction', 1) == 0 and
            len(reasons) == expected_restarts and
            all(reason == 'no_improvement' for reason in reasons))


def calibration_diagnosis(checks, full, original, expected_restarts):
    full_ok = calibrated(full, expected_restarts)
    original_ok = calibrated(original, expected_restarts)
    audit = checks['decomposition']
    if not audit['all_coalitions_pass'] or not audit['float32_aggregation_pass'] or not audit['float64_aggregation_pass']:
        stage = 'decomposition'
    elif not checks['checkpoint_data_gradient_equal'] or not checks['flatten_unpack_roundtrip_equal']:
        stage = 'gradient_input_setup'
    elif not checks['full_idlg_label_correct'] and full.get('preset') == 'idlg':
        stage = 'label_inference'
    elif original_ok and not full_ok:
        stage = 'aggregation_sensitivity_or_optimization_basin'
    elif full_ok and not original_ok:
        stage = 'inconsistent_positive_controls'
    elif not full_ok:
        stage = 'reconstruction_calibration'
    else:
        stage = 'ready_for_coalitions'
    return dict(calibrated=stage == 'ready_for_coalitions', stage=stage,
                full_calibrated=full_ok, original_calibrated=original_ok, target_mse=.01,
                full_private_mse=full['evaluation']['private_mse'],
                original_private_mse=original['evaluation']['private_mse'],
                full_vs_original_reconstruction_mse=None,
                note='Controls use identical attack configuration and seeds; no oracle image initialization.')
