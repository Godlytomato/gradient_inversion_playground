import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def gradient_magnitude(
    package: grad_package,
    context: transformation_context
) -> list[torch.Tensor]:

    return [
        torch.abs(grad)
        for grad in package.grads
    ]