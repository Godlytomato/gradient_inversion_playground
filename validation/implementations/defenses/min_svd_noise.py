from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def min_svd_noise(
    percent: float = 0.1,
    scale: float = 0.01,
    iterations: int = 5,
    solve_iterations: int = 8,
    damping: float = 1e-4,
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

    if solve_iterations <= 0:
        raise ValueError(
            "Number of solve iterations must be positive"
        )

    if damping <= 0:
        raise ValueError(
            "Damping must be greater than zero"
        )

    return [
        grad_transform(
            name="min_svd_noise",

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
                        "parameters": {
                            "iterations": iterations,
                            "solve_iterations": solve_iterations,
                            "damping": damping,
                        },
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


MIN_SVD_NOISE_PRESET = defense_preset_def(
    name="min_svd_noise",
    builder=min_svd_noise,
)
