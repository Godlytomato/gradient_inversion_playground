from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def laplace_noise(
    relative_scale: float = 0.01,
    percent: float = 1.0,
) -> list[grad_transform]:

    if relative_scale < 0:
        raise ValueError(
            "Noise scale cannot be negative"
        )

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Selection percentage must be between 0 and 1"
        )

    return [
        grad_transform(
            name="laplace_noise",

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
                        "name": "relative_gradient_scale",
                        "parameters": {
                            "relative_scale": relative_scale,
                            "mode": "global",
                        },
                    },
                },
            ),
        )
    ]


LAPLACE_NOISE_PRESET = defense_preset_def(
    name="laplace_noise",
    builder=laplace_noise,
)