from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def large_gradient_pruning(
    percent: float = 0.1,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    return [
        grad_transform(
            name="large_gradient_pruning",

            ranking_policy=function_package(
                name="gradient_magnitude",
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


LARGE_GRADIENT_PRUNING_PRESET = defense_preset_def(
    name="large_gradient_pruning",
    builder=large_gradient_pruning,
)