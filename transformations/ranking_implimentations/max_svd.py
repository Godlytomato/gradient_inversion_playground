"""Coordinate scores from a checked dominant singular triplet."""
import torch
from transformations.transform_def import grad_package, transformation_context
from .transform_utils import max_singular_direction, reshape_scores


def max_svd(
    package: grad_package,
    context: transformation_context,
    iterations: int | None = None,
    tolerance: float = 1e-4,
    exact_max_elements: int = 200000,
    exact_max_input_size: int = 64,
    max_iterations: int | None = None,
) -> list[torch.Tensor]:
    """Rank absolute left coordinates, sharing the constructor's estimator."""
    _, direction, _ = max_singular_direction(
        context, iterations=iterations, tolerance=tolerance,
        exact_max_elements=exact_max_elements, exact_max_input_size=exact_max_input_size,
        max_iterations=max_iterations,
    )
    return reshape_scores(direction.abs(), package)
