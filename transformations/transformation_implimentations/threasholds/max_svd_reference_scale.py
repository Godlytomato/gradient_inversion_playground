
import math

import torch
from transformations.ranking_implimentations.transform_utils import record_numerical_diagnostic

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from transformations.transformation_implimentations.max_svd_alignment import (
    max_svd_alignment,
)

from transformations.transformation_implimentations.threasholds.lower_bound_threashold import (
    lower_bound_threshold,
)


def max_svd_reference_scale(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    target_error: float | None = None,
    mu_j: float | None = None,
    reference_iterations: int | None = None,
    reference_tolerance: float = 1e-4,
    reference_max_iterations: int | None = None,
    **kwargs
) -> list[torch.Tensor]:
    if target_error is None:
        target_error = context.target_mse

    if target_error is None:
        raise ValueError("No reconstruction error target configured")
    if not math.isfinite(target_error) or target_error < 0:
        raise ValueError("target_error (MSE) must be finite and nonnegative")
    if (not package.grads or len(package.grads) != len(noise_tensors)
            or len(noise_tensors) != len(selection_masks)):
        raise ValueError("Gradient, noise, and mask lists must match and be nonempty")

    selected_noise = []
    for gradient, noise, mask in zip(package.grads, noise_tensors, selection_masks):
        if gradient.shape != noise.shape or noise.shape != mask.shape:
            raise ValueError("Gradient, noise, and mask shapes must match")
        selected = torch.where(mask.bool(), noise, torch.zeros_like(noise))
        if not torch.isfinite(selected).all():
            raise ValueError("Selected reference-scaled noise must be finite")
        selected_noise.append(selected)

    diagnostic = {"status": "constructing_reference", "certified": False}
    record_numerical_diagnostic(context, "max_svd_reference_scale", diagnostic)
    # The required perturbation is exactly zero for a zero target, regardless
    # of the Jacobian. A numerical direction solve cannot improve this result.
    if target_error == 0:
        diagnostic.update(status="zero_target", reference_l2=0.0, applied_l2=0.0)
        return [torch.zeros_like(noise) for noise in selected_noise]

    # construct the max-SVD reference direction
    try:
        reference_noise = max_svd_alignment(
            package,
            selection_masks,
            context,
            iterations=reference_iterations,
            tolerance=reference_tolerance,
            max_iterations=reference_max_iterations,
        )
    except Exception as error:
        diagnostic.update(status="reference_direction_error", error_type=type(error).__name__,
                          message=str(error))
        raise

    # calculate the noise required for its lower bound
    diagnostic["status"] = "scaling_reference_bound"
    try:
        reference_noise = lower_bound_threshold(
            package,
            reference_noise,
            selection_masks,
            context,
            target_error=target_error,
            mu_j=mu_j,
            **kwargs
        )
    except Exception as error:
        diagnostic.update(status="reference_bound_error", error_type=type(error).__name__,
                          message=str(error))
        raise

    # reference noise magnitude
    reference_norm = torch.linalg.vector_norm(
        torch.cat([
            noise.reshape(-1).to(torch.float64)
            for noise in reference_noise
        ])
    )

    # magnitude of the test noise
    noise_norm = torch.linalg.vector_norm(
        torch.cat([
            noise.reshape(-1).to(torch.float64)
            for noise in selected_noise
        ])
    )

    if not torch.isfinite(reference_norm) or not torch.isfinite(noise_norm):
        diagnostic["status"] = "nonfinite_norm"
        raise ValueError("Reference and selected noise norms must be finite")
    if noise_norm == 0:
        if reference_norm > 0:
            diagnostic["status"] = "zero_test_direction"
            raise ValueError(
                "Cannot scale zero noise to a nonzero reference"
            )
        record_numerical_diagnostic(context, "max_svd_reference_scale", {
            "status": "zero_reference_and_test", "reference_l2": 0.0,
            "applied_l2": 0.0, "certified": False,
        })
        return selected_noise

    # Normalize before multiplying, in float64: an otherwise valid tiny test
    # direction can require a multiplier that overflows its original dtype.
    scaled_noise = [((noise.to(torch.float64) / noise_norm) * reference_norm).to(noise.dtype)
                    for noise in selected_noise]
    if any(not torch.isfinite(noise).all() for noise in scaled_noise):
        diagnostic.update(status="nonfinite_scaled_noise", reference_l2=float(reference_norm))
        raise ValueError("Reference-scaled noise cannot be represented finitely in its original dtype")
    applied_norm = torch.linalg.vector_norm(torch.cat([
        noise.reshape(-1).to(torch.float64) for noise in scaled_noise
    ]))
    diagnostic.update(
        status="matched_reference_norm", reference_l2=float(reference_norm),
        applied_l2=float(applied_norm), test_multiplier=float(reference_norm / noise_norm),
        reference_bound_status=getattr(context, "numerical_diagnostics", {}).get("lower_bound", {}).get("status"),
    )
    return scaled_noise
