from run_def import reconstruction_job, function_package

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)


def fedleak() -> reconstruction_job:

    return reconstruction_job(
        attack_type="optimization",

        initialization_policy="random",
        label_policy="known",

        restarts=3,

        optimizer_description=function_package(
            name="adam",
            parameters={
                "lr": 0.1,
            },
        ),

        loss=function_package(
            name="partial_gradient_matching",
            parameters={
                "percent": 0.1,
            },
        ),

        missingness_policy="visible_only",
        preprocessing_gradients_policy="none",

        loss_mods=[],
        regularizers=[],

        post_update_mods=[
            function_package(
                name="clamp",
                parameters={
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            ),
        ],

        pre_update_mods=[
            function_package(
                name="fedleak_gradient_regularization",
                parameters={
                    "perturbation_size": 0.01,
                    "blend": 0.5,
                },
            ),
        ],

        scheduler=None,
        base_seed=0,
    )


FEDLEAK_PRESET = attack_preset_def(
    name="fedleak",
    builder=fedleak,
    max_iterations=1000,
    faithful=False,
)