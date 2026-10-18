# reconstruction/optimization/labels/implementations/idlg.py

import torch

from transformations.transform_def import grad_package
from reconstruction.optimization.labels.label_policy_def import label_state

def idlg_labels(
    gradients: grad_package,
) -> torch.Tensor:

    # assumes the final gradient tensor is the classifier bias gradient
    bias_gradient = gradients.grads[-1]

    if bias_gradient.ndim != 1:
        raise ValueError(
            "iDLG requires the final classifier bias gradient"
        )

    label = torch.argmin(bias_gradient)

    return label_state(
        labels=label.reshape(1),
        trainable=False
    )