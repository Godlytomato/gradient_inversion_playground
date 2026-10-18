import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def gaussian(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context
) -> list[torch.Tensor]:

    noise = []

    for grad in package.grads:
        noise.append(
            torch.randn_like(grad)
        )

    return noise