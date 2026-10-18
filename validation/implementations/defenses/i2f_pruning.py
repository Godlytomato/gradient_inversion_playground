from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def i2f_pruning(
    percent: float = 0.1,
    samples: int = 10,
    damping: float = 1e-4,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    if samples <= 0:
        raise ValueError(
            "Number of samples must be positive"
        )

    if damping <= 0:
        raise ValueError(
            "Damping must be greater than zero"
        )

    return [
        grad_transform(
            name="i2f_pruning",

            ranking_policy=function_package(
                name="exact_i2f",
                parameters={
                    "samples": samples,
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
                name="zero",
            ),
        )
    ]


I2F_PRUNING_PRESET = defense_preset_def(
    name="i2f_pruning",
    builder=i2f_pruning,
)