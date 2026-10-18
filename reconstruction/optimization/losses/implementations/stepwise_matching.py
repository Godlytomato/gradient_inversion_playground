import torch


def stepwise_matching_loss(
    reconstructed_gradients,
    target_gradients,
    state=None,
    step_size=1,
    dropout_rate=0.0,
) -> torch.Tensor:

    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    if len(reconstructed_gradients) == 0:
        raise ValueError(
            "Gradient lists cannot be empty"
        )

    if step_size <= 0:
        raise ValueError(
            "step_size must be greater than 0"
        )

    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(
            "dropout_rate must be in the range [0, 1)"
        )

    if state is None:
        active_count = len(
            reconstructed_gradients
        )

    else:
        active_count = min(
            len(reconstructed_gradients),
            1 + (
                state.iteration
                // step_size
            )
        )

    reconstructed_active = reconstructed_gradients[
        -active_count:
    ]

    target_active = target_gradients[
        -active_count:
    ]

    losses = []

    for reconstructed, target in zip(
        reconstructed_active,
        target_active
    ):

        reconstructed_flat = reconstructed.reshape(-1)
        target_flat = target.reshape(-1)

        if dropout_rate > 0.0:

            mask = (
                torch.rand_like(
                    reconstructed_flat
                )
                >= dropout_rate
            )

            if not mask.any():
                continue

            reconstructed_flat = reconstructed_flat[
                mask
            ]

            target_flat = target_flat[
                mask
            ]

        losses.append(
            torch.mean(
                (
                    reconstructed_flat
                    - target_flat
                ) ** 2
            )
        )

    if not losses:
        return reconstructed_gradients[0].new_tensor(
            0.0,
            requires_grad=True,
        )

    return torch.stack(losses).mean()