from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def dgp(
    percent: float = 0.1,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    return [
        grad_transform(
            name="dgp",

            ranking_policy=function_package(
                name="gradient_magnitude",
            ),

            draft_policy=function_package(
                name="highest_and_lowest",
                parameters={
                    "percent": percent,
                },
            ),

            transform=function_package(
                name="zero",
            ),
        )
    ]


DGP_PRESET = defense_preset_def(
    name="dgp",
    builder=dgp,
)