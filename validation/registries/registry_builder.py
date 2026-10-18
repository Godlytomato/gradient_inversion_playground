import importlib
import pkgutil

from validation.implementations.preset_def import preset_def

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)
from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)
from validation.implementations.models.model_def import (
    model_preset_def,
)
from validation.implementations.datasets.dataset_def import (
    dataset_preset_def,
)


def discover_presets(package_name, expected_type):

    package = importlib.import_module(package_name)
    registry = {}

    for module_info in pkgutil.iter_modules(package.__path__):

        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(
            f"{package_name}.{module_info.name}"
        )

        for value in vars(module).values():

            if not isinstance(value, expected_type):
                continue

            # Only collect presets defined in this module,
            # not presets imported from another module.
            if value.__class__ is not expected_type:
                continue

            if value.name in registry:
                raise ValueError(
                    f"Duplicate preset name: {value.name}"
                )

            registry[value.name] = value

    return registry


def build_registries():

    return {
        "attacks": discover_presets(
            "validation.implementations.attacks",
            attack_preset_def,
        ),

        "defenses": discover_presets(
            "validation.implementations.defenses",
            defense_preset_def,
        ),

        "models": discover_presets(
            "validation.implementations.models",
            model_preset_def,
        ),

        "datasets": discover_presets(
            "validation.implementations.datasets",
            dataset_preset_def,
        ),
    }