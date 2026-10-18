import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from .transform_utils import (
    estimated_jacobian_magnitude,
    reshape_scores,
)


def jacobian_mag(
    package: grad_package,
    context: transformation_context,
    samples: int = 10,
) -> list[torch.Tensor]:
    """
    Estimate the Jacobian row magnitude for each
    gradient coordinate using randomized probes.
    """

    flat_scores = estimated_jacobian_magnitude(
        context=context,
        samples=samples,
    )

    return reshape_scores(
        flat_scores,
        package,
    )