import torch


def visible_nonzero(
    reconstructed_gradients,
    target_gradients,
    missing_mask=None,
    threshold: float = 0.0,
):
    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Gradient list lengths do not match"
        )

    if missing_mask is not None and len(missing_mask) != len(target_gradients):
        raise ValueError(
            "Missing mask length does not match gradients"
        )

    if threshold < 0:
        raise ValueError(
            "Threshold must be nonnegative"
        )

    filtered_reconstructed = []
    filtered_target = []

    for i, (reconstructed, target) in enumerate(
        zip(reconstructed_gradients, target_gradients)
    ):
        if reconstructed.shape != target.shape:
            raise ValueError(
                f"Gradient shape mismatch at index {i}"
            )

        visible = torch.ones_like(
            target,
            dtype=torch.bool
        )

        # visible_only part
        if missing_mask is not None:
            if missing_mask[i].shape != target.shape:
                raise ValueError(
                    f"Mask shape mismatch at index {i}"
                )
            if missing_mask[i].dtype != torch.bool:
                raise TypeError(
                    f"Mask must be boolean at index {i}"
                )
            if missing_mask[i].device != target.device:
                raise ValueError(
                    f"Mask device mismatch at index {i}"
                )

            visible = visible & ~missing_mask[i]

        # ignore zeros part
        visible = visible & (target.abs() > threshold)

        filtered_reconstructed.append(
            reconstructed[visible]
        )
        filtered_target.append(
            target[visible]
        )

    return (
        filtered_reconstructed,
        filtered_target,
    )