from dataclasses import dataclass
from typing import Callable

import torch

@dataclass
class ranking_policy:
    name: str
    func: Callable[..., list[torch.Tensor]]

@dataclass
class draft_policy:
    name: str
    func: Callable[..., list[torch.Tensor]]
