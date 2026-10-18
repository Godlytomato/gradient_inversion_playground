import torch

from transformations.transform_def import grad_package


def zero(
    package: grad_package,
    selection_masks: list[torch.Tensor]
) -> grad_package:

    # initialize missing mask if this is the first pruning
    if package.missing_mask is None:
        package.missing_mask = [
            torch.zeros_like(grad, dtype=torch.bool)
            for grad in package.grads
        ]

    for grad, mask, missing in zip(
        package.grads,
        selection_masks,
        package.missing_mask
    ):
        grad[mask] = 0
        missing[mask] = True

    return package