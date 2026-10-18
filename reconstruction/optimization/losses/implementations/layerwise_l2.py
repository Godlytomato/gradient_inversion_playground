import torch


def layerwise_l2_loss(
    reconstructed_gradients,
    target_gradients,
    state=None,
) -> torch.Tensor:

    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    if len(reconstructed_gradients) == 0:
        raise ValueError(
            "Gradient lists cannot be empty"
        )

    losses = []

    for reconstructed, target in zip(
        reconstructed_gradients,
        target_gradients
    ):
        losses.append(
            torch.mean(
                (reconstructed - target) ** 2
            )
        )

    return torch.stack(losses).mean()