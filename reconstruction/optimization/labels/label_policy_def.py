from dataclasses import dataclass

import torch

from reconstruction.attack_def import component_def


@dataclass
class label_policy_def(component_def):
    pass


@dataclass
class label_state:
    labels: torch.Tensor
    trainable: bool = False


def validate_label_state(value) -> label_state:

    if not isinstance(value, label_state):
        raise TypeError(
            "Label policies must return label_state"
        )

    if not isinstance(value.labels, torch.Tensor):
        raise TypeError(
            "label_state.labels must be a torch.Tensor"
        )

    return value