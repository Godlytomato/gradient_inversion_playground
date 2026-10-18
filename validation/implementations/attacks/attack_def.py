from dataclasses import dataclass
from typing import Callable

from run_def import reconstruction_job
from validation.implementations.preset_def import preset_def


@dataclass(frozen=True)
class attack_preset_def(preset_def):

    builder: Callable[..., reconstruction_job]

    max_iterations: int = 1000
    faithful: bool = False

    def build(self, **parameters) -> reconstruction_job:
        return self.builder(**parameters)   