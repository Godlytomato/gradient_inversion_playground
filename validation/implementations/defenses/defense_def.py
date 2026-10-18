from dataclasses import dataclass
from typing import Callable

from run_def import grad_transform
from validation.implementations.preset_def import preset_def


@dataclass(frozen=True)
class defense_preset_def(preset_def):

    builder: Callable[..., list[grad_transform]]

    def build(self, **parameters) -> list[grad_transform]:
        return self.builder(**parameters)