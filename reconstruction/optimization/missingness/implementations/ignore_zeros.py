import torch


def ignore_zeros(
    reconstructed_gradients,
    target_gradients,
    missing_mask=None,
    threshold: float = 0.0,
):
    if threshold < 0:
        raise ValueError("Threshold must be nonnegative")

    filtered_reconstructed = []
    filtered_target = []

    for i, (reconstructed, target) in enumerate(
        zip(reconstructed_gradients, target_gradients)
    ):
        # Ignore zero-valued observed gradients
        visible = target.abs() > threshold

        # Also respect existing missing coordinates
        if missing_mask is not None:
            visible = visible & ~missing_mask[i]

        filtered_reconstructed.append(reconstructed[visible])
        filtered_target.append(target[visible])

    return filtered_reconstructed, filtered_target