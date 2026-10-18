from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def max_svd_lower_bound(
    percent: float = 0.1,
    target_error: float = 0.02,
    mu_j: float = 0.1,
    iterations: int | None = None,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Selection percentage must be between 0 and 1"
        )

    if target_error < 0:
        raise ValueError(
            "Target error cannot be negative"
        )

    if mu_j <= 0:
        raise ValueError(
            "mu_j must be greater than zero"
        )

    if iterations is not None and iterations <= 0:
        raise ValueError(
            "Number of iterations must be positive"
        )

    return [
        grad_transform(
            name="max_svd_lower_bound",

            ranking_policy=function_package(
                name="max_svd",
                parameters={
                    "iterations": iterations,
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
                        "name": "max_svd_alignment",
                        "parameters": {"iterations": iterations},
                    },
                    "scaling": {
                        "name": "lower_bound_threshold",
                        "parameters": {
                            "target_error": target_error,
                            "mu_j_method": "manual",
                            "mu_j_debug": False,
                            "mu_j": mu_j,
                        },
                    },
                },
            ),
        )
    ]


MAX_SVD_LOWER_BOUND_PRESET = defense_preset_def(
    name="max_svd_lower_bound",
    builder=max_svd_lower_bound,
)
