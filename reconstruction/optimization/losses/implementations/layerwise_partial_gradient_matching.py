import torch


def layerwise_partial_gradient_matching(
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

    losses = []

    for reconstructed, target in zip(
        reconstructed_gradients,
        target_gradients
    ):
        reconstructed_flat = reconstructed.reshape(-1)
        target_flat = target.reshape(-1)

        count = max(
            1,
            int(target_flat.numel() * percent)
        )

        indices = torch.topk(
            target_flat.abs(),
            k=count,
            largest=True,
            sorted=False,
        ).indices

        difference = (
            reconstructed_flat[indices]
            - target_flat[indices]
        )

        losses.append(
            torch.mean(
                difference ** 2
            )
        )

    return torch.stack(losses).mean()