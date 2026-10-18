from run_def import reconstruction_job, function_package

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)


def geiping() -> reconstruction_job:

    return reconstruction_job(
        attack_type="optimization",

        initialization_policy="random",
        label_policy="known",

        restarts=8,

        optimizer_description=function_package(
            name="adam",
            parameters={
                "lr": 0.1,
            },
        ),

        loss="cosine",

        missingness_policy="visible_only",
        preprocessing_gradients_policy="none",

        loss_mods=[],

        regularizers=[
            function_package(
                name="tv",
                parameters={
                    "weight": 1e-7,
                },
            ),
        ],

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
                name="sign",
            ),
        ],

        scheduler=None,
        base_seed=0,
    )


GEIPING_PRESET = attack_preset_def(
    name="geiping",
    builder=geiping,
    max_iterations=4800,
    faithful=True,
)