"""Scale a gradient-space blur using the reconstruction-error lower bound.

IMPORTANT: ``max_scale`` is the MAXIMUM RELATIVE L2 perturbation, NOT an
absolute multiplier. For nonzero g0, the absolute scale limit is
    max_absolute_scale = max_scale * ||g0||_2.
Thus max_scale=0.2 means a blur with L2 norm at most 20% of ||g0||_2.
The default max_scale=1000.0 is retained for backward compatibility, but
is a very large budget: choose a smaller value in your configuration.

The function signature and return type (list[torch.Tensor]) are unchanged.
On an INFEASIBLE target, it emits a RuntimeWarning and returns the best-effort
blur within the budget (the scale maximizing the bound's feasibility margin).
THIS DOES NOT GUARANTEE THE REQUESTED ERROR. Check the warning/logs.
An empirical estimate of mu_j or sigma is not a mathematical certificate.

Conventions: J_g maps input-space perturbations to gradient-space perturbations;
this implementation uses ||J_g^T d|| for gradient-space direction d. Ensure
that this matches the Jacobian in the theoretical inequality.
"""

import math
import warnings

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.ranking_implimentations.transform_utils import (
    batch_tensors,
    max_singular_direction,
    jacobian_transpose_vector_product,
    record_numerical_diagnostic,
)
from transformations.ranking_implimentations.jacobian_variation import estimate_mu_j


def lower_bound_threshold(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    target_error: float,
    mu_j: float | None = None,
    mu_j_method: str = "empirical",
    mu_j_radii: tuple[float, ...] = (1e-3, 1e-2),
    mu_j_points: int = 4,
    mu_j_directions: int = 4,
    mu_j_safety_factor: float = 1.5,
    mu_j_seed: int = 0,
    mu_j_debug: bool = False,
    initial_scale: float = 0.001,
    growth_factor: float = 2.0,
    max_scale: float = 1000.0,
    refine_iterations: int = 40,
    svd_iterations: int | None = None,
    svd_tolerance: float = 1e-4,
    svd_max_iterations: int | None = None,
) -> list[torch.Tensor]:
    """Return a masked blur of minimal sufficient norm, or a bounded best effort.

    ``initial_scale`` and ``growth_factor`` are accepted for compatibility;
    the bounded concave solver does not require exponential scale growth.
    """
    if not math.isfinite(target_error) or target_error < 0:
        raise ValueError("target_error (MSE) must be finite and nonnegative")
    if (not math.isfinite(initial_scale) or initial_scale <= 0
            or not math.isfinite(growth_factor) or growth_factor <= 1
            or not math.isfinite(max_scale) or max_scale < 0
            or refine_iterations < 1):
        raise ValueError("Invalid scale-search settings")
    if (len(package.grads) != len(noise_tensors)
            or len(noise_tensors) != len(selection_masks)
            or not package.grads):
        raise ValueError("Gradient, noise, and mask lists must match and be nonempty")

    diagnostic = {
        "status": "initializing", "target_mse": target_error,
        "max_relative_l2": max_scale, "mu_j_method": mu_j_method,
        "certified": False,
    }
    record_numerical_diagnostic(context, "lower_bound", diagnostic)

    def numerical_stage(name, calculation):
        diagnostic["status"] = name
        try:
            return calculation()
        except Exception as error:
            diagnostic.update(status="numerical_error", failed_stage=name,
                              error_type=type(error).__name__, message=str(error))
            raise

    selected_noise = []
    for grad, noise, mask in zip(package.grads, noise_tensors, selection_masks):
        if grad.shape != noise.shape or noise.shape != mask.shape:
            raise ValueError("Gradient, noise, and mask shapes must match")
        selected_noise.append(torch.where(mask.bool(), noise, torch.zeros_like(noise)))

    def zeros():
        return [torch.zeros_like(n) for n in selected_noise]

    if target_error == 0:
        diagnostic.update(status="zero_target", applied_l2=0.0)
        return zeros()

    # Scalar geometry in float64 for more stable dot products and norms.
    d0 = torch.cat([n.reshape(-1).detach().to(torch.float64)
                    for n in selected_noise])
    dnorm = torch.linalg.vector_norm(d0).item()
    if not math.isfinite(dnorm) or dnorm == 0:
        diagnostic.update(status="zero_or_invalid_selected_direction", applied_l2=0.0)
        warnings.warn("lower_bound_threshold: zero/invalid selected blur; returning zero blur",
                      RuntimeWarning, stacklevel=2)
        return zeros()

    g0 = torch.cat([g.reshape(-1).detach().to(torch.float64)
                    for g in package.grads])
    gnorm = torch.linalg.vector_norm(g0).item()
    if not math.isfinite(gnorm):
        raise ValueError("Original gradients have nonfinite norm")

    # Absolute L2 budget derived from the RELATIVE parameter max_scale.
    budget = max_scale * gnorm
    if not math.isfinite(budget):
        raise ValueError("Relative L2 budget overflowed")
    if budget == 0:
        diagnostic.update(status="zero_budget", applied_l2=0.0, gradient_norm=gnorm)
        warnings.warn("lower_bound_threshold: zero perturbation budget; returning zero blur",
                      RuntimeWarning, stacklevel=2)
        return zeros()

    # Keep d in the original gradient dtype/device for the framework's J^T v.
    normalized_noise = [n / dnorm for n in selected_noise]
    d = torch.cat([n.reshape(-1) for n in normalized_noise]).detach()
    d64 = torch.cat([n.reshape(-1).detach().to(torch.float64)
                     for n in normalized_noise])

    images, _ = batch_tensors(context)
    target_l2 = math.sqrt(float(images.numel()) * target_error)
    if not math.isfinite(target_l2):
        raise ValueError("target MSE converted to nonfinite L2 error")

    if mu_j_method == "empirical":
        if mu_j_debug:
            print("=== MU_J RADIUS DIAGNOSTICS ===")
            for radius in (1e-4, 1e-3, 1e-2, 1e-1):
                estimate = estimate_mu_j(
                    context, radii=(radius,), points=mu_j_points,
                    directions=mu_j_directions,
                    safety_factor=mu_j_safety_factor, seed=mu_j_seed,
                )
                print(f"radius={radius:.0e}, mu_j={float(estimate):.8g}")
        mu = float(numerical_stage("estimating_jacobian_variation", lambda: estimate_mu_j(
            context, radii=mu_j_radii, points=mu_j_points,
            directions=mu_j_directions,
            safety_factor=mu_j_safety_factor, seed=mu_j_seed,
        )))
    elif mu_j_method == "manual":
        if mu_j is None:
            raise ValueError("manual mu_j_method requires mu_j")
        mu = float(mu_j)
    else:
        raise ValueError("mu_j_method must be 'empirical' or 'manual'")
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu_j must be finite and nonnegative")

    diagnostic.update(mu_j=mu, target_l2=target_l2,
                      gradient_norm=gnorm, absolute_budget=budget)
    sigma_value, _, _ = numerical_stage("estimating_spectral_norm", lambda: max_singular_direction(
        context, iterations=svd_iterations, tolerance=svd_tolerance,
        max_iterations=svd_max_iterations,
    ))
    sigma = float(sigma_value)
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid estimated Jacobian spectral norm")

    gain_value = numerical_stage("estimating_direction_effect", lambda: jacobian_transpose_vector_product(d, context))
    gain = float(torch.linalg.vector_norm(gain_value.detach().to(torch.float64)))
    if not math.isfinite(gain) or gain < 0:
        raise ValueError("Invalid directional Jacobian gain")

    # Reverse sign if it reduces ||g0 + alpha*d|| without changing ||J^T*d||.
    original_alignment = float(torch.dot(g0, d64))
    if original_alignment > 0:
        normalized_noise = [-n for n in normalized_noise]
        d64 = -d64
    alignment = float(torch.dot(g0, d64))
    # Use ||g0||^2 - <g0,d>^2, with cancellation guard.
    perpendicular_sq = max(0.0, (gnorm - abs(alignment)) * (gnorm + abs(alignment)))

    def gradient_norm_at(scale: float) -> float:
        return math.hypot(math.sqrt(perpendicular_sq), alignment + scale)

    def residual(scale: float) -> float:
        value = scale * gain - target_l2 * (
            sigma * sigma + sigma * mu * target_l2
            + 2.0 * mu * gradient_norm_at(scale)
        )
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite lower-bound residual")
        return value

    # F is concave on [0, budget]. Golden-section search does not need
    # derivatives and remains stable at the norm's nondifferentiable point.
    left, right = 0.0, budget
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - phi * (right - left)
    x2 = left + phi * (right - left)
    f1, f2 = residual(x1), residual(x2)
    for _ in range(max(80, refine_iterations + 40)):
        if f1 < f2:
            left = x1
            x1, f1 = x2, f2
            x2 = left + phi * (right - left)
            f2 = residual(x2)
        else:
            right = x2
            x2, f2 = x1, f1
            x1 = right - phi * (right - left)
            f1 = residual(x1)
    candidates = ((0.0, residual(0.0)), (budget, residual(budget)),
                  (x1, f1), (x2, f2), ((left+right)/2, residual((left+right)/2)))
    peak, peak_residual = max(candidates, key=lambda pair: pair[1])

    print("=== LOWER BOUND THRESHOLD ===")
    print(f"mu_j={mu:.8g} sigma_max={sigma:.8g} target_l2={target_l2:.8g}")
    print(f"direction_effect={gain:.8g} gradient_norm={gnorm:.8g}")
    print(f"max_relative_l2={max_scale:.8g} absolute_budget={budget:.8g}")
    print(f"peak_scale={peak:.8g} peak_residual={peak_residual:.8g}")

    # No certificate available at any scale in the permitted interval.
    # Returning a best effort retains the SAME return type and prevents run crashes.
    if peak_residual < 0:
        warnings.warn(
            "lower_bound_threshold: INFEASIBLE under current constants and "
            f"relative L2 budget ({max_scale:.6g}); returning best-effort blur "
            f"of relative L2 {peak/gnorm:.6g}. This does NOT certify "
            f"the target MSE {target_error:.6g}.",
            RuntimeWarning, stacklevel=2,
        )
        scale = peak
        diagnostic["status"] = "infeasible_best_effort"
        print("status=infeasible_best_effort")
    else:
        # Start at zero (negative residual for any positive target) and bisect
        # on the increasing portion, which ends at the peak.
        low, high = 0.0, peak
        for _ in range(max(refine_iterations, 52)):
            mid = (low + high) / 2.0
            if residual(mid) >= 0:
                high = mid
            else:
                low = mid
        scale = high
        diagnostic["status"] = "feasible_under_supplied_constants"
        print("status=feasible_under_supplied_constants")

    print(f"selected_scale={scale:.8g} relative_l2={scale/gnorm:.8g}")
    print("=============================")
    diagnostic.update(
        mu_j=mu, sigma_max_estimate=sigma, target_l2=target_l2,
        direction_effect=gain, gradient_norm=gnorm, absolute_budget=budget,
        peak_scale=peak, peak_residual=peak_residual, applied_l2=scale,
        relative_l2=scale / gnorm, residual=residual(scale),
        sigma_method=getattr(context, "numerical_diagnostics", {}).get("max_svd", {}).get("method"),
    )
    return [n * scale for n in normalized_noise]
