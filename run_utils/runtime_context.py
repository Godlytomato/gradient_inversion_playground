from dataclasses import dataclass, field
from typing import Any

import torch

from data.dataset_def import dataset_def
from models.model_def import model_definition
from transformations.transform_def import grad_package


@dataclass
class runtime_context:
    model: torch.nn.Module
    model_definition: model_definition
    dataset_definition: dataset_def
    batch: Any
    gradients: grad_package
    transformation_metrics: dict[str, float] = field(default_factory=dict)
    experiment_metadata: dict[str, Any] = field(default_factory=dict)
