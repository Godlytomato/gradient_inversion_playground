import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def in_place(
    package: grad_package,
    context: transformation_context
) -> list[torch.Tensor]:

    return [
        torch.ones_like(grad)
        for grad in package.grads
    ]