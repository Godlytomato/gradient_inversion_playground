import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def random(
    package: grad_package,
    context: transformation_context
) -> list[torch.Tensor]:

    return [
        torch.rand_like(grad)
        for grad in package.grads
    ]