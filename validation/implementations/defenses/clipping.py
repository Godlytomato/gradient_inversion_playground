from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def clipping(
    bound: float = 1.0,
    mode: str = "norm",
) -> list[grad_transform]:

    if bound <= 0:
        raise ValueError(
            "Clipping bound must be greater than zero"
        )

    if mode not in {"norm", "elementwise"}:
        raise ValueError(
            "Clipping mode must be 'norm' or 'elementwise'"
        )

    return [
        grad_transform(
            name="clipping",

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
                        "name": "clipping_delta",
                        "parameters": {
                            "bound": bound,
                            "mode": mode,
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


CLIPPING_PRESET = defense_preset_def(
    name="clipping",
    builder=clipping,
)