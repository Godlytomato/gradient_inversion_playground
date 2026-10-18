
from dataclasses import asdict
from itertools import product
from pathlib import Path

import yaml

from run_def import (
    run_desc,
    function_package,
    observer_description,
    logging_description,
    best_selection_description,
)

from validation.registries.registry_builder import build_registries

from validation.validator import (
    resolve_selection,
    validate_compatibility,
    validate_run,
)


# ---------------------------------------------------------
# Construct one run description
# ---------------------------------------------------------

def generate_run(
    attack_preset,
    defense_preset,
    model_preset,
    dataset_preset,
    target,
) -> run_desc:

    attack = attack_preset.build()
    defense = defense_preset.build()
    model = model_preset.build()
    data = dataset_preset.build()

    name = (
        f"{attack_preset.name}"
        f"__{defense_preset.name}"
        f"__{model_preset.name}"
        f"__{dataset_preset.name}"
    )

    return run_desc(
        name=name,
        model=model,
        data=data,
        gradient_transformations=defense,
        attack=attack,

        attack_name=attack_preset.name,
        defense_name=defense_preset.name,

        observer=observer_description(
            success_policies=[
                function_package(
                    name="target_mse",
                    parameters={
                        "threshold": target,
                    },
                ),
            ],

            fail_policies=[
                function_package(
                    name="no_improvement",
                    parameters={
                        "patience": 200,
                    },
                ),
                function_package(
                    name="max_iterations",
                    parameters={
                        "limit": 100000,
                    },
                ),
            ],

            metrics=[
                function_package(name="mse"),
                function_package(name="psnr"),
                function_package(name="ssim"),
            ],

            logging=logging_description(
                logging_type="database",
                metrics=[
                    "mse",
                    "psnr",
                    "ssim",
                    "attack_loss",
                ],
            ),

            best_selection=best_selection_description(
                metric="mse",
                direction="minimize",
            ),
        ),
    )


# ---------------------------------------------------------
# Save run description to YAML
# ---------------------------------------------------------

def save_yaml(
    description: run_desc,
    output_directory: Path,
) -> Path:

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_directory / f"{description.name}.yaml"

    with path.open("w") as file:
        yaml.safe_dump(
            asdict(description),
            file,
            sort_keys=False,
        )

    return path


# ---------------------------------------------------------
# Generate complete validation matrix
# ---------------------------------------------------------

def generate_yamls(
    attacks: list[str] | None = None,
    defenses: list[str] | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    target: float | None = 0.02,
) -> list[Path]:

    if target is None:
        target = 0.02

    if not 0 <= target < float("inf"):
        raise ValueError(
            "Target MSE must be finite and nonnegative"
        )

    # Build current registries automatically.
    registries = build_registries()

    selected_attacks = resolve_selection(
        attacks, registries["attacks"]
    )

    selected_defenses = resolve_selection(
        defenses, registries["defenses"]
    )

    selected_models = resolve_selection(
        models, registries["models"]
    )

    selected_datasets = resolve_selection(
        datasets, registries["datasets"]
    )

    descriptions = []

    # Construct all combinations.
    for attack, defense, model, dataset in product(
        selected_attacks.values(),
        selected_defenses.values(),
        selected_models.values(),
        selected_datasets.values(),
    ):

        validate_compatibility(
            attack, defense, model, dataset
        )

        description = generate_run(
            attack,
            defense,
            model,
            dataset,
            target,
        )

        validate_run(description)

        descriptions.append(description)

    # Validate the complete matrix before writing files.
    names = [description.name for description in descriptions]

    if len(names) != len(set(names)):
        raise ValueError(
            "Generated runs contain duplicate names"
        )

    output_directory = Path("schedual")

    paths = []

    for description in descriptions:
        path = save_yaml(
            description,
            output_directory,
        )
        paths.append(path)

    print(f"Generated {len(paths)} YAML files")

    return paths
