from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def jacobian_pruning(
    percent: float = 0.1,
    samples: int = 10,
) -> list[grad_transform]:

    if not 0.0 <= percent <= 1.0:
        raise ValueError(
            "Pruning percentage must be between 0 and 1"
        )

    if samples <= 0:
        raise ValueError(
            "Number of Jacobian samples must be positive"
        )

    return [
        grad_transform(
            name="jacobian_pruning",

            ranking_policy=function_package(
                name="jacobian_magnitude",
                parameters={
                    "samples": samples,
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


JACOBIAN_PRUNING_PRESET = defense_preset_def(
    name="jacobian_pruning",
    builder=jacobian_pruning,
)