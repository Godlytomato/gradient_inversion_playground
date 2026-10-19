
from run_def import grad_transform, function_package
from run_utils.svd_settings import min_svd_parameters
from run_utils.bound_settings import DEFAULT_BOUND_METHOD, bound_parameters

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def min_svd_noise_reference(
    target_error: float = 0.02,
    mu_j: float = 0.1,
    percent: float = 0.1,
    iterations: int | None = None,
    solve_iterations: int | None = None,
    damping: float | None = None,
    bound_method: str = DEFAULT_BOUND_METHOD,
    residual_l2: float = 0.0,
    residual_fraction: float = 0.1,
    solver_method: str | None = None,
    krylov_max_iterations: int | None = None,
    krylov_check_every: int | None = None,
    krylov_max_elements: int | None = None,
    solver_seed: int | None = None,
) -> list[grad_transform]:

    if not 0.0 < percent <= 1.0:
        raise ValueError(
            "Selection percentage must be between 0 and 1"
        )

    if target_error <= 0:
        raise ValueError(
            "Target error must be positive"
        )

    calibration_parameters = bound_parameters(
        target_error=target_error, mu_j=mu_j, bound_method=bound_method,
        residual_l2=residual_l2, residual_fraction=residual_fraction,
    )

    solver_parameters = min_svd_parameters(
        iterations=iterations, solve_iterations=solve_iterations, damping=damping,
        solver_method=solver_method, krylov_max_iterations=krylov_max_iterations,
        krylov_check_every=krylov_check_every, krylov_max_elements=krylov_max_elements,
        solver_seed=solver_seed,
    )

    return [
        grad_transform(
            name="min_svd_noise_reference",

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
                name="noise",
                parameters={
                    "construction": {
                        "name": "min_svd_alignment",
                        "parameters": solver_parameters.copy(),
                    },
                    "scaling": {
                        "name": "max_svd_reference_scale",
                        "parameters": calibration_parameters,
                    },
                },
            ),
        )
    ]


MIN_SVD_NOISE_REFERENCE_PRESET = defense_preset_def(
    name="min_svd_noise_reference",
    builder=min_svd_noise_reference,
)
