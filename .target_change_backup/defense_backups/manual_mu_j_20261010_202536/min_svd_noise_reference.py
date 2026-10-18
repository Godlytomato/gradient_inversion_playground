
from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def min_svd_noise_reference(
    target_error: float = 0.0001,
    mu_j: float = 1.0,
    percent: float = 0.1,
    iterations: int = 5,
    solve_iterations: int = 8,
    damping: float = 1e-4,
) -> list[grad_transform]:

    if not 0.0 < percent <= 1.0:
        raise ValueError(
            "Selection percentage must be between 0 and 1"
        )

    if target_error <= 0:
        raise ValueError(
            "Target error must be positive"
        )

    if mu_j <= 0:
        raise ValueError(
            "mu_j must be positive"
        )

    if iterations <= 0 or solve_iterations <= 0:
        raise ValueError(
            "Iterations must be positive"
        )

    if damping <= 0:
        raise ValueError(
            "Damping must be positive"
        )

    return [
        grad_transform(
            name="min_svd_noise_reference",

            ranking_policy=function_package(
                name="min_svd",
                parameters={
                    "iterations": iterations,
                    "solve_iterations": solve_iterations,
                    "damping": damping,
                },
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
                        "parameters": {},
                    },
                    "scaling": {
                        "name": "max_svd_reference_scale",
                        "parameters": {
                            "target_error": target_error,
                            "mu_j_method": "empirical",
                            "mu_j_debug": True,
                            "mu_j": mu_j,
                        },
                    },
                },
            ),
        )
    ]


MIN_SVD_NOISE_REFERENCE_PRESET = defense_preset_def(
    name="min_svd_noise_reference",
    builder=min_svd_noise_reference,
)
