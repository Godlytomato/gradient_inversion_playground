
from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def uniform_noise_reference(
    target_error: float = 0.02,
    mu_j: float = 0.1,
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
            name="uniform_noise_reference",

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
                        "name": "uniform",
                        "parameters": {},
                    },
                    "scaling": {
                        "name": "max_svd_reference_scale",
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


UNIFORM_NOISE_REFERENCE_PRESET = defense_preset_def(
    name="uniform_noise_reference",
    builder=uniform_noise_reference,
)
