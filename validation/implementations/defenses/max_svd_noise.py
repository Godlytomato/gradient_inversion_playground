from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def max_svd_noise(
    percent: float = 0.1,
    scale: float = 0.01,
    iterations: int = 10,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Selection percentage must be between 0 and 1"
        )

    if scale < 0:
        raise ValueError(
            "Noise scale cannot be negative"
        )

    if iterations <= 0:
        raise ValueError(
            "Number of iterations must be positive"
        )

    return [
        grad_transform(
            name="max_svd_noise",

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
                        "parameters": {},
                    },
                    "scaling": {
                        "name": "preset",
                        "parameters": {
                            "scale": scale,
                        },
                    },
                },
            ),
        )
    ]


MAX_SVD_NOISE_PRESET = defense_preset_def(
    name="max_svd_noise",
    builder=max_svd_noise,
)