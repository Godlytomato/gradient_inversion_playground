import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def uniform(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context
) -> list[torch.Tensor]:

    noise = []

    for grad in package.grads:
        noise.append(
            torch.rand_like(grad) * 2 - 1
        )

    return noise