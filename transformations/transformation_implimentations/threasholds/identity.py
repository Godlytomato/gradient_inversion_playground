# transformations/transformation_implimentations/threasholds/identity.py

import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def identity(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    **kwargs
) -> list[torch.Tensor]:

    return noise_tensors