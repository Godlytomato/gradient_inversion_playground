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


def validate_label_state(value, batch_size=None, class_count=None, device=None) -> label_state:

    if not isinstance(value, label_state):
        raise TypeError(
            "Label policies must return label_state"
        )

    if not isinstance(value.labels, torch.Tensor):
        raise TypeError(
            "label_state.labels must be a torch.Tensor"
        )

    if not isinstance(value.trainable, bool):
        raise TypeError("label_state.trainable must be a bool")

    labels = value.labels
    if device is not None and labels.device != torch.device(device):
        raise ValueError("Labels and reconstruction must be on the same device")

    if value.trainable:
        if labels.ndim != 2 or not labels.is_floating_point():
            raise ValueError("Trainable labels must be floating logits with shape [batch, classes]")
        if not labels.is_leaf or not labels.requires_grad:
            raise ValueError("Trainable label logits must be leaf tensors requiring gradients")
        if not torch.isfinite(labels).all():
            raise ValueError("Trainable label logits must be finite")
        if class_count is not None and labels.shape[1] != class_count:
            raise ValueError("Trainable label class count does not match the model")
    else:
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("Fixed labels must be a one-dimensional torch.long tensor")
        if labels.numel() and (labels < 0).any():
            raise ValueError("Fixed labels must be nonnegative class indices")
        if class_count is not None and labels.numel() and (labels >= class_count).any():
            raise ValueError("Fixed label class index is outside the model output range")

    if batch_size is not None and labels.shape[0] != batch_size:
        raise ValueError("Label batch size does not match reconstruction batch size")

    return value
