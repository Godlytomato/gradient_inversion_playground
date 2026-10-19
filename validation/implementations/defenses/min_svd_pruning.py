from run_def import grad_transform, function_package
from run_utils.svd_settings import min_svd_parameters

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def min_svd_pruning(
    percent: float = 0.1,
    iterations: int | None = None,
    solve_iterations: int | None = None,
    damping: float | None = None,
    solver_method: str | None = None,
    krylov_max_iterations: int | None = None,
    krylov_check_every: int | None = None,
    krylov_max_elements: int | None = None,
    solver_seed: int | None = None,
    rank_policy: str | None = None,
    relative_rank_tolerance: float | None = None,
    dense_max_elements: int | None = None,
    dense_max_input_size: int | None = None,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    solver_parameters = min_svd_parameters(
        iterations=iterations, solve_iterations=solve_iterations, damping=damping,
        solver_method=solver_method, krylov_max_iterations=krylov_max_iterations,
        krylov_check_every=krylov_check_every, krylov_max_elements=krylov_max_elements,
        solver_seed=solver_seed, rank_policy=rank_policy,
        relative_rank_tolerance=relative_rank_tolerance,
        dense_max_elements=dense_max_elements, dense_max_input_size=dense_max_input_size,
    )

    return [
        grad_transform(
            name="min_svd_pruning",

            ranking_policy=function_package(
                name="min_svd",
                parameters=solver_parameters.copy(),
            ),

            draft_policy=function_package(
                name="highest_draft",
                parameters={
                    "percent": percent,
                },
            ),

            transform=function_package(
                name="zero",
            ),
        )
    ]


MIN_SVD_PRUNING_PRESET = defense_preset_def(
    name="min_svd_pruning",
    builder=min_svd_pruning,
)