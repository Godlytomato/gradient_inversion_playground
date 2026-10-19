import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from transformations.ranking_implimentations.transform_utils import (
    min_singular_direction,
)

from transformations.transformation_implimentations.tranform_utils import (
    selected_indices,
    vector_to_noise,
)


def min_svd_alignment(
    package: grad_package,
    selection_masks: list[torch.Tensor],
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

    indices = selected_indices(
        package,
        selection_masks,
    )

    if indices.numel() == 0:
        return [
            torch.zeros_like(gradient)
            for gradient in package.grads
        ]

    # u is the singular direction in gradient space
    _, u, _ = min_singular_direction(
        context, iterations=iterations, solve_iterations=solve_iterations,
        damping=damping, tolerance=tolerance, exact_max_elements=exact_max_elements,
        exact_max_input_size=exact_max_input_size, rank_policy=rank_policy,
        solver_method=solver_method, krylov_max_iterations=krylov_max_iterations,
        krylov_check_every=krylov_check_every, krylov_max_elements=krylov_max_elements,
        solver_seed=solver_seed,
    )

    direction = u[
        indices
    ]

    norm = torch.linalg.vector_norm(
        direction, dtype=torch.float64,
    )

    if norm != 0:
        direction = direction / norm

    return vector_to_noise(
        package,
        selection_masks,
        direction,
    )
