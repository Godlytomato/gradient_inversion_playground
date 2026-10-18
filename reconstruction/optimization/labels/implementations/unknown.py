# reconstruction/optimization/labels/implementations/unknown.py

import torch

from reconstruction.optimization.labels.label_policy_def import label_state


def unknown_labels(
    batch_size: int,
    class_count: int,
    device
) -> torch.Tensor:

    return label_state(
        labels=torch.randn(
            batch_size,
            class_count,
            device=device,
            requires_grad=True
        ),
        trainable=True
    )