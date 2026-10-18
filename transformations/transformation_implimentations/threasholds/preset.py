import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def preset(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    scale: float
) -> list[torch.Tensor]:

    return [
        noise * scale
        for noise in noise_tensors
    ]