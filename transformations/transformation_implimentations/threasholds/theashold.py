import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def threshold(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    threshold: float
) -> list[torch.Tensor]:

    # norm only over values that will actually receive noise
    squared_sum = 0.0

    for noise, mask in zip(
        noise_tensors,
        selection_masks
    ):
        mask = mask.bool()
        squared_sum += torch.sum(
            noise[mask] ** 2
        )

    norm = torch.sqrt(squared_sum)

    # avoid divide by zero
    if norm == 0:
        return noise_tensors

    scale = threshold / norm

    return [
        noise * scale
        for noise in noise_tensors
    ]