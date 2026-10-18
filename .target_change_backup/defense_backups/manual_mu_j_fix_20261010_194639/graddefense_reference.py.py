
from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def graddefense_reference(
    slices_num: int = 10,
    perturb_slices_num: int = 2,
    target_error: float = 0.0001,
    mu_j: float = 1.0,
    seed: int = 0,
) -> list[grad_transform]:

    if slices_num <= 0:
        raise ValueError(
            "Number of slices must be greater than zero"
        )

    if not 0 <= perturb_slices_num <= slices_num:
        raise ValueError(
            "Invalid number of perturbation slices"
        )

    if target_error <= 0:
        raise ValueError(
            "Target error must be positive"
        )

    if mu_j <= 0:
        raise ValueError(
            "mu_j must be positive"
        )

    return [
        grad_transform(
            name="graddefense_reference",

            ranking_policy=function_package(
                name="graddefense_sensitivity",
            ),

            draft_policy=function_package(
                name="sensitivity_slices",
                parameters={
                    "slices_num": slices_num,
                    "perturb_slices_num": perturb_slices_num,
                    "seed": seed,
                },
            ),

            transform=function_package(
                name="noise",
                parameters={
                    "construction": {
                        "name": "gaussian",
                        "parameters": {},
                    },
                     "scaling": {
                        "name": "max_svd_reference_scale",
                        "parameters": {
                            "mu_j_method": "manual",
                            "target_error": 0.0001,
                            "mu_j_method": "manual",
                            "mu_j": 1.0,
                        },
                    },
                },
            ),
        )
    ]


GRADDEFENSE_REFERENCE_PRESET = defense_preset_def(
    name="graddefense_reference",
    builder=graddefense_reference,
)
