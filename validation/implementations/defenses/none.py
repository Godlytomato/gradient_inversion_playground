from run_def import grad_transform

from validation.implementations.defenses.defense_def import (
    defense_preset_def,
)


def no_defense() -> list[grad_transform]:

    return []


NONE_PRESET = defense_preset_def(
    name="none",
    builder=no_defense,
)