from run_def import reconstruction_job, function_package

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)


def dlg() -> reconstruction_job:

    return reconstruction_job(
        attack_type="optimization",

        initialization_policy="random",
        label_policy="unknown",

        restarts=1,

        optimizer_description=function_package(
            name="lbfgs",
            parameters={
                "lr": 1.0,
                "max_iter": 20,
            },
        ),

        loss="l2",

        missingness_policy="visible_nonzero",
        preprocessing_gradients_policy="none",

        loss_mods=[],
        regularizers=[],

        post_update_mods=[
        ],

        pre_update_mods=[],

        scheduler=None,
        base_seed=0,
    )


DLG_PRESET = attack_preset_def(
    name="dlg",
    builder=dlg,
    max_iterations=300,
    faithful=True,
)