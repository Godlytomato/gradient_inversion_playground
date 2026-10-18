from dataclasses import dataclass, field
from typing import Callable, Any

import torch


# package that contains the gradients as they move through transformations
@dataclass
class grad_package:
    grads: list[torch.Tensor]
    missing_mask: list[torch.Tensor] | None = None

@dataclass
class transformation:
    name: str
    func: Callable[..., grad_package]

@dataclass
class transformation_context:
    model: torch.nn.Module
    batch: Any

    labels: torch.Tensor | None = None
    loss_function: Callable | None = None
    device: torch.device | None = None
    target_mse: float | None = None
    numerical_diagnostics: dict[str, Any] = field(default_factory=dict)
