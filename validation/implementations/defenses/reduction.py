from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def reduction(
    factor: float = 0.5,
) -> list[grad_transform]:

    if not 0.0 <= factor <= 1.0:
        raise ValueError(
            "Reduction factor must be between 0 and 1"
        )

    return [
        grad_transform(
            name="reduction",

            ranking_policy=function_package(
                name="in_place",
            ),

            draft_policy=function_package(
                name="highest_draft",
                parameters={
                    "percent": 1.0,
                },
            ),

            transform=function_package(
                name="noise",
                parameters={
                    "construction": {
                        "name": "reduction",
                        "parameters": {
                            "factor": factor,
                        },
                    },
                    "scaling": {
                        "name": "identity",
                        "parameters": {},
                    },
                },
            ),
        )
    ]


REDUCTION_PRESET = defense_preset_def(
    name="reduction",
    builder=reduction,
)