"""Empirical local Jacobian-Lipschitz estimator (not an upper-bound certificate)."""
import math
import torch
from transformations.ranking_implimentations.transform_utils import (
    batch_tensors, model_gradient_vector, record_numerical_diagnostic,
)


def estimate_mu_j(context, radii=(1e-3, 1e-2), points=4, directions=4, safety_factor=1.5, seed=0):
    if not radii or any(not math.isfinite(r) or r <= 0 for r in radii):
        raise ValueError("radii must contain positive values")
    if points < 1 or directions < 1 or not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError("points and directions must be positive; safety_factor >= 1")

    images, labels = batch_tensors(context)
    images = images.detach()
    generator = torch.Generator(device=images.device).manual_seed(seed)

    def jvp_at(x, v):
        def gradient_map(z):
            return model_gradient_vector(
                context.model, z, labels, loss_function=context.loss_function,
                create_graph=True,
            )
        return torch.autograd.functional.jvp(
            gradient_map, x, v, create_graph=False, strict=False
        )[1]

    largest = 0.0
    for radius in radii:
        for _ in range(points):
            displacement = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
            displacement = radius * displacement / displacement.norm().clamp_min(1e-30)
            shifted = images + displacement
            for _ in range(directions):
                v = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
                v = v / v.norm().clamp_min(1e-30)
                # Measure Jacobian variation before dividing by radius.
                difference = (
                    jvp_at(shifted, v) - jvp_at(images, v)
                ).norm().item()

                ratio = difference / radius

                print(
                    f"radius={radius:.1e}, "
                    f"jacobian_difference={difference:.8g}, "
                    f"ratio={ratio:.8g}"
                )
                if not math.isfinite(ratio):
                    raise ValueError("Nonfinite Jacobian-variation estimate")
                largest = max(largest, ratio)
    diagnostic = {
        "method": "empirical_directional_probes", "radii": list(radii),
        "points": points, "directions": directions, "seed": seed,
        "safety_factor": safety_factor, "observed_maximum": largest,
        "estimate": largest * safety_factor, "is_certified_upper_bound": False,
        "status": "estimated" if largest else "zero_observed_variation",
    }
    record_numerical_diagnostic(context, "jacobian_variation", diagnostic)
    # Zero is legitimate for an affine gradient map. It is an observation,
    # not proof of a region-wide zero Lipschitz constant (especially at gates).
    return largest * safety_factor
