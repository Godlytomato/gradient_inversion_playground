
from run_def import grad_transform, function_package
from run_utils.bound_settings import DEFAULT_BOUND_METHOD, bound_parameters

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def laplace_noise_reference(
    target_error: float = 0.02,
    mu_j: float = 0.1,
    percent: float = 1.0,
    bound_method: str = DEFAULT_BOUND_METHOD,
    residual_l2: float = 0.0,
    residual_fraction: float = 0.1,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
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

    return [
        grad_transform(
            name="laplace_noise_reference",

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
                        "name": "laplace",
                        "parameters": {},
                    },
                    "scaling": {
                        "name": "max_svd_reference_scale",
                        "parameters": calibration_parameters,
                    },
                },
            ),
        )
    ]


LAPLACE_NOISE_REFERENCE_PRESET = defense_preset_def(
    name="laplace_noise_reference",
    builder=laplace_noise_reference,
)
