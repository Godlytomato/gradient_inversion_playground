"""Conditional, residual-aware calibration of gradient perturbations.

The default directional Taylor bound allows nonzero gradient-matching residual.
It uses one J_g.T*d product and no maximum-SVD solve. Uniform smoothness and
the declared residual condition remain assumptions, not certified facts for a
ReLU model. Post-attack evaluation uses the measured residual separately.

``max_scale`` is a relative L2 budget: max_scale*||original gradients||.
Infeasible targets return bounded best effort with explicit diagnostics. The
previous exact-fit formula is available only as ``legacy_proof_v1``.
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


def _legacy_lower_bound_threshold(
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
        "certified": False, "bound_method": "legacy_proof_v1",
        "requires_exact_gradient_match": True,
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


def lower_bound_threshold(
    package, noise_tensors, selection_masks, context, target_error,
    mu_j=None, mu_j_method="empirical", mu_j_radii=(1e-3, 1e-2),
    mu_j_points=4, mu_j_directions=4, mu_j_safety_factor=1.5,
    mu_j_seed=0, mu_j_debug=False, initial_scale=0.001, growth_factor=2.0,
    max_scale=1000.0, refine_iterations=40, svd_iterations=None,
    svd_tolerance=1e-4, svd_max_iterations=None,
    bound_method="directional_residual_v1", residual_l2=0.0,
    residual_fraction=0.1,
):
    """Conditional calibration allowing a nonzero raw gradient residual.

    With a=||eta||, k=||Dg(x0).T*(eta/a)|| and E=||xhat-x0||,
    a-eps <= k*E + mu_J*E**2/2. Calibrate reconstructions satisfying
    eps <= residual_l2 + residual_fraction*a; this is an explicit condition,
    not a promise that the attack meets it. Uniform smoothness/constants are
    NOT certified for arbitrary models, especially ReLU networks.

    This method needs one VJP, not a maximum singular-direction solve.
    Old search/SVD keywords are retained for explicit legacy compatibility.
    """
    from transformations.reconstruction_bound import required_noise_l2, residual_aware_bound

    if bound_method == "legacy_proof_v1":
        return _legacy_lower_bound_threshold(
            package, noise_tensors, selection_masks, context, target_error,
            mu_j, mu_j_method, mu_j_radii, mu_j_points, mu_j_directions,
            mu_j_safety_factor, mu_j_seed, mu_j_debug, initial_scale,
            growth_factor, max_scale, refine_iterations, svd_iterations,
            svd_tolerance, svd_max_iterations,
        )
    if bound_method != "directional_residual_v1":
        raise ValueError(f"Unknown lower-bound method: {bound_method}")
    # Validate residual conditions even for trivial targets/directions.
    required_noise_l2(0., 1., 0., residual_l2, residual_fraction)
    if not math.isfinite(target_error) or target_error < 0:
        raise ValueError("target_error (MSE) must be finite and nonnegative")
    if not math.isfinite(max_scale) or max_scale < 0:
        raise ValueError("max_scale must be finite and nonnegative")
    if (not package.grads or len(package.grads) != len(noise_tensors)
            or len(noise_tensors) != len(selection_masks)):
        raise ValueError("Gradient, noise, and mask lists must match and be nonempty")
    diagnostic = {
        "status": "initializing", "bound_method": bound_method,
        "target_mse": target_error, "max_relative_l2": max_scale,
        "residual_l2": residual_l2, "residual_fraction": residual_fraction,
        "residual_condition": "raw_gradient_l2 <= residual_l2 + residual_fraction * noise_l2",
        "certified": False, "mu_j_method": mu_j_method,
        "uniform_smoothness_verified": False,
        "residual_condition_verified": False,
        "nonsmooth_module_types": sorted({type(m).__name__ for m in context.model.modules()
            if isinstance(m, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.LeakyReLU,
                              torch.nn.PReLU, torch.nn.MaxPool1d,
                              torch.nn.MaxPool2d, torch.nn.MaxPool3d))}),
    }
    record_numerical_diagnostic(context, "lower_bound", diagnostic)
    selected = []
    for grad, noise, mask in zip(package.grads, noise_tensors, selection_masks):
        if grad.shape != noise.shape or noise.shape != mask.shape:
            raise ValueError("Gradient, noise, and mask shapes must match")
        value = torch.where(mask.bool(), noise, torch.zeros_like(noise))
        if not torch.isfinite(value).all() or not torch.isfinite(grad).all():
            raise ValueError("Selected noise and original gradients must be finite")
        selected.append(value)
    zeros = lambda: [torch.zeros_like(n) for n in selected]
    if target_error == 0:
        diagnostic.update(status="zero_target", applied_l2=0.0)
        return zeros()
    original = getattr(context, "original_gradients", None)
    if original is not None and any(not torch.equal(g, g0) for g, g0 in zip(package.grads, original)):
        raise ValueError("Conditional bound calibration must precede other gradient modifications")
    diagnostic["baseline_verified"] = original is not None
    norm = float(torch.linalg.vector_norm(torch.cat([n.reshape(-1).to(torch.float64) for n in selected])))
    gnorm = float(torch.linalg.vector_norm(torch.cat([g.reshape(-1).to(torch.float64) for g in package.grads])))
    if not math.isfinite(norm) or not math.isfinite(gnorm):
        raise ValueError("Noise and gradient norms must be finite")
    budget = max_scale * gnorm
    if not math.isfinite(budget):
        raise ValueError("Relative L2 budget overflowed")
    if norm == 0 or budget == 0:
        diagnostic.update(status="zero_direction" if norm == 0 else "zero_budget", applied_l2=0.0)
        warnings.warn("Conditional bound: zero direction/budget; target not calibrated", RuntimeWarning, stacklevel=2)
        return zeros()
    direction = [(n.to(torch.float64) / norm).to(n.dtype) for n in selected]
    images, _ = batch_tensors(context)
    radius = math.sqrt(images.numel() * target_error)
    diagnostic.update(target_l2=radius, gradient_norm=gnorm, absolute_budget=budget)

    def stage(name, function):
        diagnostic["status"] = name
        try:
            return function()
        except Exception as error:
            diagnostic.update(status="numerical_error", failed_stage=name,
                              error_type=type(error).__name__, message=str(error))
            raise

    if mu_j_method == "manual":
        if mu_j is None:
            raise ValueError("manual mu_j_method requires mu_j")
        mu = float(mu_j)
    elif mu_j_method == "empirical":
        mu = float(stage("estimating_jacobian_variation", lambda: estimate_mu_j(
            context, radii=mu_j_radii, points=mu_j_points, directions=mu_j_directions,
            safety_factor=mu_j_safety_factor, seed=mu_j_seed,
        )))
    else:
        raise ValueError("mu_j_method must be 'empirical' or 'manual'")
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("mu_j must be finite and nonnegative")
    diagnostic["mu_j"] = mu
    effect = stage("estimating_directional_gain", lambda: jacobian_transpose_vector_product(
        torch.cat([n.reshape(-1) for n in direction]).detach(), context))
    gain = float(torch.linalg.vector_norm(effect.to(torch.float64)))
    required = required_noise_l2(radius, gain, mu, residual_l2, residual_fraction)
    diagnostic.update(direction_effect=gain, required_l2=required if required is None or math.isfinite(required) else None)
    if required is None:
        diagnostic.update(status="degenerate_constants", applied_l2=0.0)
        warnings.warn("Conditional bound: zero gain and variation cannot calibrate positive error", RuntimeWarning, stacklevel=2)
        return zeros()
    scale = min(required, budget)
    output = [(n.to(torch.float64) * scale).to(n.dtype) for n in direction]
    if any(not torch.isfinite(n).all() for n in output):
        raise ValueError("Calibrated noise cannot be represented in its original dtype")
    applied = float(torch.linalg.vector_norm(torch.cat([n.reshape(-1).to(torch.float64) for n in output])))
    eps_budget = residual_l2 + residual_fraction * applied
    bound = residual_aware_bound(applied, eps_budget, gain, mu)
    lower = bound["lower_bound_l2"]
    calibrated = required <= budget and lower is not None and lower >= radius * (1 - 8 * torch.finfo(output[0].dtype).eps)
    diagnostic.update(
        status="conditional_target_calibrated" if calibrated else "infeasible_best_effort",
        applied_l2=applied, relative_l2=applied / gnorm,
        assumed_residual_l2=eps_budget, conditional_lower_bound_l2=lower,
        conditional_lower_bound_mse=None if lower is None else lower * lower / images.numel(),
        calibration_target_met=calibrated,
        rounding_relative_tolerance=8 * torch.finfo(output[0].dtype).eps,
    )
    if not calibrated:
        warnings.warn("Conditional bound: target INFEASIBLE within perturbation budget; using bounded best effort", RuntimeWarning, stacklevel=2)
    print(f"[BOUND] method={bound_method} status={diagnostic['status']} mu_j={mu:.6g} "
          f"gain={gain:.6g} applied_l2={applied:.6g} certified=False", flush=True)
    return output
