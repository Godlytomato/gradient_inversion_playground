"""Scale the actually selected perturbation to a relative L2 budget."""

import math
import torch

from transformations.transform_def import grad_package, transformation_context


def relative_gradient_scale(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    relative_scale: float,
    mode: str = "layer",
    **kwargs,
) -> list[torch.Tensor]:
    if not math.isfinite(relative_scale) or relative_scale < 0:
        raise ValueError("relative_scale must be finite and non-negative")
    if mode not in {"layer", "global"}:
        raise ValueError(f"Unknown relative scaling mode: {mode}")
    if not package.grads or not len(package.grads) == len(noise_tensors) == len(selection_masks):
        raise ValueError("Gradient, noise, and mask lists must match and be nonempty")
    selected_noise = []
    for gradient, noise, mask in zip(package.grads, noise_tensors, selection_masks):
        if gradient.shape != noise.shape or mask.shape != noise.shape:
            raise ValueError("Gradient, noise, and mask shapes must match")
        selected_noise.append(torch.where(mask.bool(), noise, torch.zeros_like(noise)))

    def scaled(gradients, noises):
        grad_norm = torch.linalg.vector_norm(torch.cat([g.detach().reshape(-1).to(torch.float64) for g in gradients]))
        noise_norm = torch.linalg.vector_norm(torch.cat([n.detach().reshape(-1).to(torch.float64) for n in noises]))
        if not bool(torch.isfinite(grad_norm)) or not bool(torch.isfinite(noise_norm)):
            raise ValueError("Gradient and selected noise norms must be finite")
        if noise_norm == 0:
            return [torch.zeros_like(noise) for noise in noises]
        scale = relative_scale * grad_norm / noise_norm
        return [noise * scale.to(dtype=noise.dtype, device=noise.device) for noise in noises]

    if mode == "global":
        return scaled(package.grads, selected_noise)
    output = []
    for gradient, noise in zip(package.grads, selected_noise):
        output.extend(scaled([gradient], [noise]))
    return output
