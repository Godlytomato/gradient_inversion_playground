import torch


def l2_loss(
    reconstructed_gradients,
    target_gradients,
    state=None,
) -> torch.Tensor:

    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    loss = None

    for reconstructed, target in zip(
        reconstructed_gradients,
        target_gradients
    ):
        value = torch.sum(
            (reconstructed - target) ** 2
        )

        if loss is None:
            loss = value
        else:
            loss = loss + value

    if loss is None:
        raise ValueError(
            "Gradient lists cannot be empty"
        )

    return loss