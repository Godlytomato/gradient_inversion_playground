# svd_utils.py

import torch

from transformations.transform_def import grad_package


def selected_indices(
    package: grad_package,
    selection_masks: list[torch.Tensor]
) -> torch.Tensor:

    flat_masks = [
        mask.bool().reshape(-1)
        for mask in selection_masks
    ]

    full_mask = torch.cat(flat_masks)

    return torch.nonzero(
        full_mask,
        as_tuple=False
    ).flatten()


def vector_to_noise(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    vector: torch.Tensor
) -> list[torch.Tensor]:

    noise = [
        torch.zeros_like(grad)
        for grad in package.grads
    ]

    loc = 0

    for noise_tensor, mask in zip(
        noise,
        selection_masks
    ):
        mask = mask.bool()

        count = int(mask.sum().item())

        noise_tensor[mask] = vector[
            loc:loc + count
        ]

        loc += count

    return noise