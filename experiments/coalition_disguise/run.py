"""Seeded sample -> decomposition -> full reconstruction -> every proper subset."""
import csv
import json

import torch
from torchvision.utils import save_image

from .common import digest, gradient, pixel_metrics, unpack, write_json
from .attacks import attack, build_attack, config_hash, execution_config, TraceObserver
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
    for path in sorted(root.glob('*/*/summary.json'), key=lambda p: (p.parent.parent.name, p.parent.name != 'full', p.parent.name)):
        summary = json.loads(path.read_text())
        if 'evaluation' not in summary:
            continue
        folder = path.parent.parent.name
        name, _, hashed = folder.partition('_')
        records.append(dict(model=summary.get('model'), dataset=summary.get('dataset'),
            seed=summary.get('seed'), private_index=summary.get('private_index'),
            decomposition=summary.get('decomposition', 'historical_sparse'),
            residual_nnz_gap=summary.get('residual_nnz_gap'),
            max_disguise_distance_squared=summary.get('max_disguise_distance_squared'),
            preset=summary.get('preset', name),
            profile=summary.get('profile', 'historical_preset'),
            configuration=summary.get('configuration', hashed), coalition=path.parent.name,
            attack_loss=summary['best_metrics']['attack_loss'],
            iterations=summary['total_iterations'], stopping_reason=summary['reason'],
            restarts=summary['run_metrics']['restarts_started'],
            disguise_index=summary['evaluation'].get('disguise_index'),
            disguise_mse=summary['evaluation'].get('disguise_mse'),
            ground_truth_mse=summary['evaluation'].get('ground_truth_mse', summary['evaluation'].get('private_mse')),
            **{k: v for k, v in summary['evaluation'].items() if k not in {'disguise_index', 'disguise_mse', 'ground_truth_mse'}}))
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
    from .workflow import main as run_workflow
    run_workflow()


if __name__ == '__main__':
    main()
