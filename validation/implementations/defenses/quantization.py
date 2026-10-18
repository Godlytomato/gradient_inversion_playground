from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def quantization(
    bits: int = 8,
    stochastic: bool = False,
) -> list[grad_transform]:

    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2:
        raise ValueError(
            "Symmetric quantization requires an integer bit width >= 2"
        )

    return [
        grad_transform(
            name="quantization",

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
                        "name": "quantization",
                        "parameters": {
                            "mode": "bits",
                            "bits": bits,
                            "stochastic": stochastic,
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


QUANTIZATION_PRESET = defense_preset_def(
    name="quantization",
    builder=quantization,
)
