
from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def gaussian_noise_reference(
    target_error: float = 0.02,
    mu_j: float = 1.0,
    percent: float = 1.0,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
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

    return [
        grad_transform(
            name="gaussian_noise_reference",

            ranking_policy=function_package(
                name="in_place",
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
                        "name": "gaussian",
                        "parameters": {},
                    },
                    "scaling": {
                        "name": "max_svd_reference_scale",
                        "parameters": {
                            "target_error": 0.02,
                            "mu_j_method": "empirical",
                            "mu_j": mu_j,
                        },
                    },
                },
            ),
        )
    ]


GAUSSIAN_NOISE_REFERENCE_PRESET = defense_preset_def(
    name="gaussian_noise_reference",
    builder=gaussian_noise_reference,
)
