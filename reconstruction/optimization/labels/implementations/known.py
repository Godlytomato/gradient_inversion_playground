import torch

from reconstruction.optimization.labels.label_policy_def import label_state


def known_labels(labels):
    return label_state(
        labels=labels.clone(),
        trainable=False
    )