import torch


def visible_only(reconstructed_gradients, target_gradients, missing_mask):
    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError("Gradient list lengths do not match")
    if missing_mask is not None and len(missing_mask) != len(target_gradients):
        raise ValueError("Missing mask length does not match gradients")

    for i, (reconstructed, target) in enumerate(zip(reconstructed_gradients, target_gradients)):
        if reconstructed.shape != target.shape:
            raise ValueError(f"Gradient shape mismatch at index {i}")
        if missing_mask is not None:
            if missing_mask[i].shape != target.shape:
                raise ValueError(f"Mask shape mismatch at index {i}")
            if missing_mask[i].dtype != torch.bool:
                raise TypeError(f"Mask must be boolean at index {i}")
            if missing_mask[i].device != target.device:
                raise ValueError(f"Mask device mismatch at index {i}")

    if missing_mask is None:
        return reconstructed_gradients, target_gradients

    return (
        [r[~m] for r, m in zip(reconstructed_gradients, missing_mask)],
        [t[~m] for t, m in zip(target_gradients, missing_mask)],
    )
