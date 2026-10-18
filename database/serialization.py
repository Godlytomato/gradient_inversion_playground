import json
from dataclasses import asdict, is_dataclass

from database.hashing import (
    run_hash,
    attack_hash,
    defense_hash,
)


def to_dict(
    value
):
    if is_dataclass(value):
        return asdict(value)

    return value


def parameters_json(
    parameters
):
    return json.dumps(
        parameters,
        sort_keys=True,
        default=str,
    )


def serialize_function_package(
    run_id,
    category,
    package,
    position=0,
):

    if package is None:
        return None

    if isinstance(
        package,
        str,
    ):
        return {
            "run_id": run_id,
            "category": category,
            "name": package,
            "position": position,
            "parameters": parameters_json(
                {}
            ),
        }

    return {
        "run_id": run_id,
        "category": category,
        "name": package.name,
        "position": position,
        "parameters": parameters_json(
            package.parameters
        ),
    }


def serialize_attack_components(
    run_id,
    attack,
):

    components = []

    single_components = {
        "initialization":
            attack.initialization_policy,

        "label_policy":
            attack.label_policy,

        "optimizer":
            attack.optimizer_description,

        "loss":
            attack.loss,

        "missingness":
            attack.missingness_policy,

        "preprocessing_gradients":
            attack.preprocessing_gradients_policy,

        "scheduler":
            attack.scheduler,
    }

    for (
        category,
        package,
    ) in single_components.items():

        component = (
            serialize_function_package(
                run_id=run_id,
                category=category,
                package=package,
            )
        )

        if component is not None:
            components.append(
                component
            )

    multiple_components = {
        "loss_mod":
            attack.loss_mods,

        "regularizer":
            attack.regularizers,

        "post_update":
            attack.post_update_mods,

        "pre_update":
            attack.pre_update_mods,
    }

    for (
        category,
        packages,
    ) in multiple_components.items():

        for (
            position,
            package,
        ) in enumerate(
            packages
        ):

            components.append(
                serialize_function_package(
                    run_id=run_id,
                    category=category,
                    package=package,
                    position=position,
                )
            )

    if attack.gan_description is not None:

        components.append(
            serialize_function_package(
                run_id=run_id,
                category="gan",
                package=(
                    attack.gan_description
                ),
            )
        )

    return components


def serialize_defense_components(
    run_id,
    transformations,
):

    components = []

    for (
        position,
        transformation,
    ) in enumerate(
        transformations
    ):

        components.append(
            serialize_function_package(
                run_id=run_id,
                category="defense_ranking",
                package=(
                    transformation
                    .ranking_policy
                ),
                position=position,
            )
        )

        components.append(
            serialize_function_package(
                run_id=run_id,
                category="defense_draft",
                package=(
                    transformation
                    .draft_policy
                ),
                position=position,
            )
        )

        components.append(
            serialize_function_package(
                run_id=run_id,
                category="defense_transform",
                package=(
                    transformation
                    .transform
                ),
                position=position,
            )
        )

    return components


def serialize_run(
    description
):

    raw_description = json.dumps(
        to_dict(
            description
        ),
        sort_keys=True,
        default=str,
    )

    return {
        "run_hash":
            run_hash(
                description
            ),

        "name":
            description.name,

        "model":
            description.model,

        "dataset":
            description.data.dataset,

        "batch_size":
            description.data.size,

        "attack_hash":
            attack_hash(
                description
            ),

        "attack_name":
            description.attack_name,

        "defense_hash":
            defense_hash(
                description
            ),

        "defense_name":
            description.defense_name,

        "raw_description":
            raw_description,
    }