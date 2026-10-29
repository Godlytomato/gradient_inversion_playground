from dataclasses import dataclass
from typing import Callable

import torch

from reconstruction.attack_def import component_def


@dataclass
class initialization_def(component_def):
    pass


@dataclass
class parameterized_image:
    """Opt-in image coordinates; ordinary tensor initializers are unchanged."""

    variable: torch.Tensor
    decode: Callable[[torch.Tensor], torch.Tensor]
