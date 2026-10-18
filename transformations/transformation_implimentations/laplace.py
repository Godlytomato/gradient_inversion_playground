import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def laplace(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context
) -> list[torch.Tensor]:

    noise = []

    for grad in package.grads:

        distribution = torch.distributions.Laplace(
            loc=torch.tensor(
                0.0,
                device=grad.device,
                dtype=grad.dtype
            ),
            scale=torch.tensor(
                1.0,
                device=grad.device,
                dtype=grad.dtype
            )
        )

        noise.append(
            distribution.sample(grad.shape)
        )

    return noise