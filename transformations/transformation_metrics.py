import torch


def measure_transformation(original, transformed):
    delta = torch.cat([
        (new.detach() - old.detach()).reshape(-1)
        for old, new in zip(original, transformed.grads)
    ])
    original_flat = torch.cat([g.detach().reshape(-1) for g in original])

    delta_norm = torch.linalg.vector_norm(delta, dtype=torch.float64).item()
    original_norm = torch.linalg.vector_norm(original_flat, dtype=torch.float64).item()
    total = delta.numel()

    missing = (
        sum(mask.sum().item() for mask in transformed.missing_mask)
        if transformed.missing_mask is not None else 0
    )

    return {
        "delta_l2": delta_norm,
        "relative_delta_l2": delta_norm / max(original_norm, 1e-12),
        "changed_fraction": (delta != 0).sum().item() / total,
        "missing_fraction": missing / total,
    }
