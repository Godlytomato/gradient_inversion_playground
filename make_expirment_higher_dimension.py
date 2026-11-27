from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import yaml

from run_def import (
    batch,
    run_desc,
    grad_transform,
    function_package,
    observer_description,
    logging_description,
    best_selection_description,
)
from validation.implementations.attacks.gradinversion import gradinversion
from validation.validator import validate_run


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------

SERVERS = 4
BLOCK_SIZE = 32
SEED = 2026
TARGET_MSE = 0.02
OUTPUT_DIRECTORY = Path("schedual")


# ---------------------------------------------------------
# Construct one run for a compromised-server coalition
# ---------------------------------------------------------

def make_run(
    compromised_servers: int = 0,
    selection_number: int = 0,
    baseline: bool = False,
) -> run_desc:
    if baseline:
        name = "gradinversion__no_defense__resnet18__cifar10"
        transformations = []
    else:
        name = (
            f"gradinversion__ppga_s{SERVERS}"
            f"_c{compromised_servers}_sel{selection_number}"
            f"__resnet18__cifar10"
        )
        transformations = [
            grad_transform(
                name="ppga",
                ranking_policy=function_package(name="in_place"),
                draft_policy=function_package(name="all"),
                transform=function_package(
                    name="ppga",
                    parameters={
                        "servers": SERVERS,
                        "compromised_servers": compromised_servers,
                        "selection_number": selection_number,
                        "block_size": BLOCK_SIZE,
                        "seed": SEED,
                    },
                ),
            ),
        ]

    return run_desc(
        name=name,
        model="resnet18",
        data=batch(
            dataset="cifar10",
            size=1,
            classes=list(range(10)),
        ),
        gradient_transformations=transformations,
        attack=gradinversion(),
        attack_name="gradinversion",
        defense_name="none" if baseline else "ppga",
        observer=observer_description(
            success_policies=[
                function_package(
                    name="target_mse",
                    parameters={"threshold": TARGET_MSE},
                ),
            ],
            fail_policies=[
                function_package(
                    name="no_improvement",
                    parameters={"patience": 200},
                ),
                function_package(
                    name="max_iterations",
                    parameters={"limit": 100000},
                ),
            ],
            metrics=[
                function_package(name="mse"),
                function_package(name="ssim"),
            ],
            logging=logging_description(
                logging_type="file",
                metrics=["mse", "ssim"],
            ),
            best_selection=best_selection_description(
                metric="mse",
                direction="minimize",
            ),
        ),
    )


# ---------------------------------------------------------
# Generate all coalitions, full decoding, and no-defense control
# ---------------------------------------------------------

def main() -> None:
    descriptions = [(make_run(baseline=True), "undefended")]

    for compromised_servers in range(1, SERVERS + 1):
        for selection_number, coalition in enumerate(
            combinations(range(SERVERS), compromised_servers)
        ):
            descriptions.append((
                make_run(compromised_servers, selection_number),
                coalition,
            ))

    # Validate all configurations before writing any files.
    for description, _ in descriptions:
        validate_run(description)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for description, coalition in descriptions:
        path = OUTPUT_DIRECTORY / f"{description.name}.yaml"
        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(asdict(description), file, sort_keys=False)
        print(f"{path}  compromised={coalition}")

    print(f"Generated {len(descriptions)} schedules in {OUTPUT_DIRECTORY}/")


if __name__ == "__main__":
    main()