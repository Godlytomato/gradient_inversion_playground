
from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def graddefense_reference(
    slices_num: int = 10,
    perturb_slices_num: int = 2,
    target_error: float = 0.02,
    mu_j: float = 0.1,
    seed: int = 0,
    sensitivity_method: str = "frobenius_rademacher",
    sensitivity_samples: int = 16,
    sensitivity_seed: int = 0,
) -> list[grad_transform]:

    """Reference-budget variant of the declared GradDefense-inspired protocol."""
    if sensitivity_method not in {"frobenius_rademacher", "legacy_sum"}:
        raise ValueError("Unknown GradDefense sensitivity method")
    if not isinstance(sensitivity_samples, int) or isinstance(sensitivity_samples, bool) or sensitivity_samples < 1:
        raise ValueError("Sensitivity samples must be a positive integer")

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
                parameters={
                    "method": sensitivity_method,
                    "samples": sensitivity_samples,
                    "seed": sensitivity_seed,
                },
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
                            "target_error": target_error,
                            "mu_j_method": "manual",
                            "mu_j_debug": False,
                            "mu_j": mu_j,
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
