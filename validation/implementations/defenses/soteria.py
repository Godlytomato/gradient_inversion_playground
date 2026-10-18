from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def soteria(
    layer_name: str = "fc1",
    percent: float = 0.1,
    stabilizer: float = 0.1,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    if stabilizer <= 0:
        raise ValueError(
            "Stabilizer must be greater than zero"
        )

    if not layer_name:
        raise ValueError(
            "A feature layer must be specified"
        )

    return [
        grad_transform(
            name="soteria",

            ranking_policy=function_package(
                name="soteria_sensitivity",
                parameters={
                    "layer_name": layer_name,
                    "stabilizer": stabilizer,
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


SOTERIA_PRESET = defense_preset_def(
    name="soteria",
    builder=soteria,
)