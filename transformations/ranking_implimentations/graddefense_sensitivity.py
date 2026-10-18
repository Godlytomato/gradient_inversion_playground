"""Declared GradDefense-inspired layer sensitivity estimators.

The default estimates ||d g_layer / d inputs||_F using independent Rademacher
VJPs, averaging squared norms before taking a square root. ``legacy_sum`` is
retained only for explicitly labelled historical comparisons; it measures a
sum of gradient coordinates and can cancel to zero for classifiers.
"""

import math
import torch

from transformations.transform_def import grad_package, transformation_context


def graddefense_sensitivity(
    package: grad_package,
    context: transformation_context,
    method: str = "frobenius_rademacher",
    samples: int = 16,
    seed: int = 0,
    **kwargs,
) -> list[torch.Tensor]:
    if method not in {"frobenius_rademacher", "legacy_sum"}:
        raise ValueError("Unknown GradDefense sensitivity method")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("Sensitivity samples must be a positive integer")

    model = context.model
    if isinstance(context.batch, torch.Tensor):
        inputs, labels = context.batch, context.labels
    else:
        images, labels_list = [], []
        for sample in context.batch:
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise TypeError("GradDefense requires (image, label) samples")
            image, label = sample
            if not isinstance(image, torch.Tensor):
                raise TypeError("GradDefense requires image tensors")
            images.append(image)
            labels_list.append(int(label))
        if not images:
            raise ValueError("GradDefense received an empty batch")
        inputs = torch.stack(images)
        labels = torch.tensor(labels_list, dtype=torch.long)
    if context.labels is not None:
        labels = context.labels
    if labels is None:
        raise ValueError("GradDefense sensitivity requires labels")

    device = next(model.parameters()).device
    inputs = inputs.detach().clone().to(device).requires_grad_(True)
    labels = labels.to(device)
    loss_fn = context.loss_function or torch.nn.CrossEntropyLoss()
    loss = loss_fn(model(inputs), labels)
    parameters = tuple(model.parameters())
    if len(parameters) != len(package.grads):
        raise ValueError("Gradient package must match all model parameters")
    gradients = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for gradient, package_gradient in zip(gradients, package.grads):
        sensitivity = 0.0
        if gradient is not None and gradient.requires_grad:
            squared_norms = []
            for _ in range(1 if method == "legacy_sum" else samples):
                if method == "legacy_sum":
                    probe = torch.ones_like(gradient)
                else:
                    probe = torch.randint(0, 2, gradient.shape, device=device, generator=generator)
                    probe = probe.to(gradient.dtype).mul_(2).sub_(1)
                vjp = torch.autograd.grad(
                    gradient, inputs, grad_outputs=probe,
                    retain_graph=True, allow_unused=True,
                )[0]
                if vjp is None:
                    squared_norms.append(0.0)
                elif method == "legacy_sum":
                    sensitivity = float(vjp.detach().reshape(vjp.shape[0], -1).norm(dim=1).mean())
                else:
                    squared_norms.append(float(vjp.detach().to(torch.float64).square().sum()))
            if method != "legacy_sum":
                sensitivity = math.sqrt(sum(squared_norms) / samples)
        if not math.isfinite(sensitivity):
            raise ValueError("GradDefense sensitivity is not finite")
        scores.append(torch.full_like(package_gradient, sensitivity))
    return scores
