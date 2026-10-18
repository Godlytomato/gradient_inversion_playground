from run_def import reconstruction_job, function_package

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)


def hf_gradinv() -> reconstruction_job:

    return reconstruction_job(
        attack_type="optimization",

        initialization_policy="random",

        label_policy=function_package(
            name="hf_gradinv",
            parameters={
                "population_size": 200,
                "iterations": 100000,
                "feature_count": 20,
            },
        ),

        restarts=3,

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
                name="bn_stats",
                parameters={
                    "weight": 1e-6,
                },
            ),
            function_package(
                name="auxiliary_prior",
                parameters={
                    "weight": 1e-6,
                },
            ),
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
            ),
        ],

        pre_update_mods=[],

        scheduler=None,
        base_seed=0,
    )


HF_GRADINV_PRESET = attack_preset_def(
    name="hf_gradinv",
    builder=hf_gradinv,
    max_iterations=2000,
    faithful=False,
)
