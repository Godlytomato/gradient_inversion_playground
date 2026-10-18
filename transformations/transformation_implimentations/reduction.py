import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def reduction(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    factor: float,
    **kwargs
) -> list[torch.Tensor]:

    if factor < 0 or factor > 1:
        raise ValueError(
            "Reduction factor must be between 0 and 1"
        )

    deltas = []

    for gradient in package.grads:

        # target gradient after reduction
        reduced = gradient * factor

        # delta such that:
        #
        # gradient + delta = reduced gradient
        delta = reduced - gradient

        deltas.append(delta)

    return deltas