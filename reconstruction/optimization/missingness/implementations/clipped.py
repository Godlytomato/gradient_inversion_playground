import torch

from reconstruction.optimization.missingness.implementations.visible_only import (
    visible_only,
)


def clipped(
    reconstructed_gradients,
    target_gradients,
    missing_mask,
    bound=1.0,
    mode="norm",
):
    """
    Match reconstructed gradients through the clipping
    operation used by the defense.
    """

    if bound <= 0:
        raise ValueError(
            "Clipping bound must be positive"
        )

    if mode == "elementwise":

        transformed = [
            torch.clamp(
                gradient,
                min=-bound,
                max=bound,
            )
            for gradient
            in reconstructed_gradients
        ]

    elif mode == "norm":

        norm = torch.sqrt(
            sum(
                torch.sum(
                    gradient ** 2
                )
                for gradient
                in reconstructed_gradients
            )
        )

        if norm > bound:

            scale = (
                bound
                / norm
            )

            transformed = [
                gradient * scale
                for gradient
                in reconstructed_gradients
            ]

        else:

            transformed = list(
                reconstructed_gradients
            )

    else:

        raise ValueError(
            f"Unknown clipping mode: {mode}"
        )

    # Straight-through version of clipping.
    observed = [
        original
        + (
            transformed_gradient
            - original
        ).detach()

        for original, transformed_gradient
        in zip(
            reconstructed_gradients,
            transformed,
        )
    ]

    return visible_only(
        reconstructed_gradients=observed,
        target_gradients=target_gradients,
        missing_mask=missing_mask,
    )