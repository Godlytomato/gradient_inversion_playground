"""Coordinate scores from a checked minimum singular triplet."""
import torch
from transformations.transform_def import grad_package, transformation_context
from .transform_utils import min_singular_direction, reshape_scores, conjugate_gradient


def min_svd(
    package: grad_package,
    context: transformation_context,
    iterations: int | None = None,
    solve_iterations: int | None = None,
    damping: float | None = None,
    tolerance: float = 1e-4,
    exact_max_elements: int = 200000,
    exact_max_input_size: int = 64,
    rank_policy: str = "error",
    solver_method: str = "inverse_v2",
    krylov_max_iterations: int | None = None,
    krylov_check_every: int | None = None,
    krylov_max_elements: int | None = None,
    solver_seed: int | None = None,
) -> list[torch.Tensor]:
    """Rank absolute left coordinates; reject null or unconverged directions."""
    _, direction, _ = min_singular_direction(
        context, iterations=iterations, solve_iterations=solve_iterations,
        damping=damping, tolerance=tolerance, exact_max_elements=exact_max_elements,
        exact_max_input_size=exact_max_input_size, rank_policy=rank_policy,
        solver_method=solver_method, krylov_max_iterations=krylov_max_iterations,
        krylov_check_every=krylov_check_every, krylov_max_elements=krylov_max_elements,
        solver_seed=solver_seed,
    )
    return reshape_scores(direction.abs(), package)
