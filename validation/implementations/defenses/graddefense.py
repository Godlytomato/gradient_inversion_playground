from run_def import grad_transform, function_package

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def graddefense(
    slices_num: int = 10,
    perturb_slices_num: int = 2,
    relative_scale: float = 0.01,
    seed: int = 0,
    sensitivity_method: str = "frobenius_rademacher",
    sensitivity_samples: int = 16,
    sensitivity_seed: int = 0,
) -> list[grad_transform]:

    """GradDefense-inspired slicing with an explicitly recorded estimator."""
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

    if relative_scale < 0:
        raise ValueError(
            "Noise scale cannot be negative"
        )

    return [
        grad_transform(
            name="graddefense",

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
                        "name": "relative_gradient_scale",
                        "parameters": {
                            "relative_scale": relative_scale,
                            "mode": "global",
                        },
                    },
                },
            ),
        )
    ]


GRADDEFENSE_PRESET = defense_preset_def(
    name="graddefense",
    builder=graddefense,
)
