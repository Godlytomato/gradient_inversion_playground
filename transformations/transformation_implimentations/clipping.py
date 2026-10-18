import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def clipping_delta(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    bound: float,
    mode: str = "elementwise",
    **kwargs
) -> list[torch.Tensor]:

    if bound <= 0:
        raise ValueError(
            "Clipping bound must be greater than zero"
        )

    # -----------------------------------------
    # elementwise clipping
    # -----------------------------------------

    if mode == "elementwise":

        deltas = []

        for gradient in package.grads:

            clipped = torch.clamp(
                gradient,
                min=-bound,
                max=bound
            )

            # delta such that:
            #
            # gradient + delta = clipped
            delta = clipped - gradient

            deltas.append(delta)

        return deltas

    # -----------------------------------------
    # whole-gradient norm clipping
    # -----------------------------------------

    elif mode == "norm":

        total_norm = torch.sqrt(
            sum(
                torch.sum(gradient ** 2)
                for gradient in package.grads
            )
        )

        # gradient is already inside the bound
        if total_norm <= bound:
            return [
                torch.zeros_like(gradient)
                for gradient in package.grads
            ]

        scale = bound / total_norm

        deltas = []

        for gradient in package.grads:

            clipped = gradient * scale

            delta = clipped - gradient

            deltas.append(delta)

        return deltas

    else:
        raise ValueError(
            f"Unknown clipping mode: {mode}"
        )