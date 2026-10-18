import torch


def partial_gradient_matching(
    reconstructed_gradients,
    target_gradients,
    percent=0.1,
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

    if not 0.0 < percent <= 1.0:
        raise ValueError(
            "percent must be in the range (0, 1]"
        )

    reconstructed = torch.cat([
        gradient.reshape(-1)
        for gradient in reconstructed_gradients
    ])

    target = torch.cat([
        gradient.reshape(-1)
        for gradient in target_gradients
    ])

    count = max(
        1,
        int(target.numel() * percent)
    )

    indices = torch.topk(
        target.abs(),
        k=count,
        largest=True,
        sorted=False,
    ).indices

    difference = (
        reconstructed[indices]
        - target[indices]
    )

    return torch.sum(
        difference ** 2
    )