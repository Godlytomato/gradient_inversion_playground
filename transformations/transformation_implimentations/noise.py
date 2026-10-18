import torch

from transformations.transform_def import grad_package


def noise(
    package: grad_package,
    noise: list[torch.Tensor],
    selection_masks: list[torch.Tensor]
) -> grad_package:

    # add noise only to selected gradient values
    for grad, noise_tensor, mask in zip(
        package.grads,
        noise,
        selection_masks
    ):
        grad[mask] += noise_tensor[mask]

    return package