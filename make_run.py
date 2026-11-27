"""Generate the complete attack x defense x model validation sweep.

For ResNet lower-bound defenses, estimate Jacobian variation on a
Softplus-smoothed surrogate while leaving the actual attack model unchanged.
"""

from itertools import product
from pathlib import Path

from run_def import best_selection_description, function_package
from validation.generator import generate_run, save_yaml
from validation.registries.registry_builder import build_registries
from validation.validator import validate_compatibility, validate_run

TARGET_MSE = 0.02
SOFTPLUS_BETA = 10.0
RUN_TAG = "resnet_softplus10_bound_mse_stop_v2"
OUTPUT_DIR = Path("schedule")


def configure_resnet_threshold(description):
    """Use empirical mu_J on a Softplus surrogate for ResNet bounds."""
    if not description.model.startswith("resnet"):
        return

    for transformation in description.gradient_transformations:
        transform = transformation.transform
        if transform.name != "noise":
            continue
        scaling = transform.parameters.get("scaling", {})
        if scaling.get("name") != "lower_bound_threshold":
            continue

        parameters = scaling.setdefault("parameters", {})
        parameters.pop("mu_j", None)
        parameters["mu_j_method"] = "empirical"
        parameters["softplus_beta"] = SOFTPLUS_BETA


def main():
    presets = build_registries()
    generated = 0

    for attack, defense, model, dataset in product(
        presets["attacks"].values(),
        presets["defenses"].values(),
        [presets["models"]["resnet18"],presets["models"]["lenet5"]],
        [presets["datasets"]["cifar10"]],
    ):
        validate_compatibility(attack, defense, model, dataset)
        description = generate_run(
            attack, defense, model, dataset, target=TARGET_MSE
        )
        observer = description.observer
        observer.success_policies = [function_package("target_mse", {"threshold": TARGET_MSE})]
        observer.fail_policies = [
            function_package("no_improvement", {
                "metric": "mse", "patience": 200, "minimum_change": 1e-4,
            }),
            function_package("max_iterations", {"limit": attack.max_iterations}),
        ]
        observer.best_selection = best_selection_description("mse", "minimize")
        configure_resnet_threshold(description)
        description.name = f"{RUN_TAG}__{description.name}"
        validate_run(description)
        save_yaml(description, OUTPUT_DIR)
        generated += 1

    print(f"Generated {generated} runs in {OUTPUT_DIR}")
    print(f"Run with: python run.py --schedule {OUTPUT_DIR} --strict-errors")


if __name__ == "__main__":
    main()