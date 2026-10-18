
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

    # construct the max-SVD reference direction
    reference_noise = max_svd_alignment(
        package,
        selection_masks,
        context,
        iterations=reference_iterations,
        tolerance=reference_tolerance,
        max_iterations=reference_max_iterations,
    )

    # calculate the noise required for its lower bound
    reference_noise = lower_bound_threshold(
        package,
        reference_noise,
        selection_masks,
        context,
        target_error=target_error,
        mu_j=mu_j,
        **kwargs
    )

    # reference noise magnitude
    reference_norm = torch.linalg.vector_norm(
        torch.cat([
            noise.reshape(-1)
            for noise in reference_noise
        ])
    )

    # restrict test noise to selected coordinates
    selected_noise = []

    for noise, mask in zip(
        noise_tensors,
        selection_masks
    ):
        selected_noise.append(
            torch.where(
                mask.bool(),
                noise,
                torch.zeros_like(noise)
            )
        )

    # magnitude of the test noise
    noise_norm = torch.linalg.vector_norm(
        torch.cat([
            noise.reshape(-1)
            for noise in selected_noise
        ])
    )

    if noise_norm == 0:
        if reference_norm > 0:
            raise ValueError(
                "Cannot scale zero noise to a nonzero reference"
            )
        record_numerical_diagnostic(context, "max_svd_reference_scale", {
            "status": "zero_reference_and_test", "reference_l2": 0.0,
            "applied_l2": 0.0, "certified": False,
        })
        return selected_noise

    scale = reference_norm / noise_norm
    record_numerical_diagnostic(context, "max_svd_reference_scale", {
        "status": "matched_reference_norm", "reference_l2": float(reference_norm),
        "applied_l2": float(reference_norm), "test_multiplier": float(scale),
        "reference_bound_status": getattr(context, "numerical_diagnostics", {}).get("lower_bound", {}).get("status"),
        "certified": False,
    })

    return [
        noise * scale
        for noise in selected_noise
    ]
