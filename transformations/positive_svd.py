"""Checked minimum-positive singular modes with an explicit numerical rank.

The dense route stores the original-precision Jacobian, factors it in float64,
and preserves its left singular vector. Replacing that vector by a normalized
live JVP would amplify derivative roundoff near weak positive singular modes.
Both singular equations are checked independently with the public operators.
"""
import math

import torch


def _record(context, diagnostic, status, message):
    from transformations.ranking_implimentations.transform_utils import record_numerical_diagnostic

    diagnostic["status"] = status
    record_numerical_diagnostic(context, "min_svd", diagnostic)
    raise RuntimeError(message)


def _validate(context, sigma, left, right, scale, tolerance, diagnostic):
    from transformations.ranking_implimentations.transform_utils import (
        _singular_norm, jacobian_vector_product,
        jacobian_transpose_vector_product, record_numerical_diagnostic,
    )

    sigma_value, scale_value = float(sigma), float(scale)
    cutoff = float(diagnostic["relative_rank_tolerance"]) * scale_value
    if (not math.isfinite(sigma_value) or not math.isfinite(scale_value)
            or scale_value <= 0 or sigma_value <= cutoff):
        _record(context, diagnostic, "numerical_nullspace",
                "No resolved positive singular direction exceeds the declared rank cutoff")
    image = jacobian_vector_product(right, context)
    transpose = jacobian_transpose_vector_product(left, context)
    right_residual = float(_singular_norm(image - sigma * left) / scale)
    left_residual = float(_singular_norm(transpose - sigma * right) / scale)
    left_norm = float(_singular_norm(left))
    right_norm = float(_singular_norm(right))
    relative = max(right_residual, left_residual)
    diagnostic.update(
        sigma=sigma_value, spectral_scale_estimate=scale_value,
        rank_cutoff=cutoff, relative_residual=relative,
        jv_relative_residual=right_residual,
        jt_u_relative_residual=left_residual,
        left_vector_norm=left_norm, right_vector_norm=right_norm,
        norm_accumulation_dtype="float64",
    )
    if (not all(math.isfinite(value) for value in
                (relative, left_norm, right_norm))
            or relative > tolerance
            or abs(left_norm - 1.) > tolerance
            or abs(right_norm - 1.) > tolerance):
        _record(
            context, diagnostic, "unconverged",
            f"min-SVD direction did not converge: relative singular residual "
            f"{relative:.3g} exceeds {tolerance:.3g}, or singular vectors are not normalized. "
            "Increase the declared solver resources or use a bounded dense decomposition.",
        )
    diagnostic["status"] = "converged_minimum_positive_estimate"
    record_numerical_diagnostic(context, "min_svd", diagnostic)
    return sigma, left, right


def _dense(context, images, labels, settings, tolerance, diagnostic, parameter_size):
    from transformations.minimum_krylov import _derivative_products

    size = images.numel()
    probe = torch.zeros(size, device=images.device, dtype=images.dtype)
    probe[0] = 1
    jvp, _ = _derivative_products(context, images, labels, probe, diagnostic)
    # Keeping columns contiguous makes the transposed tall matrix column-major
    # for LAPACK; unlike a full SVD this never constructs a tall left basis.
    columns = torch.empty((size, parameter_size), dtype=images.dtype, device="cpu")
    for index in range(size):
        direction = torch.zeros_like(probe)
        direction[index] = 1
        column = jvp(direction)
        if column.numel() != parameter_size or not torch.isfinite(column).all():
            _record(context, diagnostic, "nonfinite_product",
                    "Invalid gradient Jacobian column in positive-SVD decomposition")
        columns[index].copy_(column.detach().to(device="cpu", dtype=images.dtype))
    triangular = torch.linalg.qr(columns.T.to(torch.float64), mode="r").R
    _, values, right_h = torch.linalg.svd(triangular, full_matrices=False)
    scale = values[0]
    cutoff = float(scale) * settings["relative_rank_tolerance"]
    rank = int((values > cutoff).sum())
    diagnostic.update(
        method="dense_qr_svd", decomposition_dtype="float64",
        jacobian_dtype=str(columns.dtype).removeprefix("torch."),
        numerical_rank=rank, rank=rank, input_nullity=size - rank,
        rank_resolution="complete_dense_jacobian", rank_complete=True,
        spectral_norm=float(scale), rank_cutoff=cutoff,
        largest_discarded_sigma=float(values[rank]) if rank < values.numel() else None,
        smallest_retained_sigma=float(values[rank - 1]) if rank else None,
        derivative_columns=size,
    )
    if not rank or not torch.isfinite(scale) or scale <= 0:
        _record(context, diagnostic, "zero_jacobian_or_no_positive_mode",
                "No positive singular direction exceeds the declared rank cutoff")
    index = rank - 1
    sigma = values[index].to(images.device)
    right_cpu = right_h[index]
    left_cpu = torch.empty(parameter_size, dtype=torch.float64)
    for first in range(0, parameter_size, 4096):
        block = columns[:, first:first + 4096].T.to(torch.float64)
        left_cpu[first:first + 4096] = (block @ right_cpu) / values[index]
    right = right_cpu.to(device=images.device, dtype=images.dtype)
    left = left_cpu.to(device=images.device, dtype=images.dtype)
    diagnostic.update(selected_index=index,
                      selected_to_cutoff_ratio=(float(values[index] / scale)
                                                / settings["relative_rank_tolerance"]),
                      left_vector_source="dense_jacobian_float64_product")
    return _validate(context, sigma, left, right, scale, tolerance, diagnostic)


def positive_singular_direction(context, images, labels, settings, tolerance):
    """Return a checked minimum-positive triplet using declared resource caps."""
    from transformations.minimum_krylov import minimum_krylov
    from transformations.ranking_implimentations.transform_utils import _singular_norm, model_gradient_vector

    size = images.numel()
    lazy_initialized = any(isinstance(parameter, torch.nn.parameter.UninitializedParameter)
                           for parameter in context.model.parameters())
    if lazy_initialized:
        # The runner initializes lazy layers during gradient production. Direct
        # API callers need the same materialization before counting parameters.
        model_gradient_vector(context.model, images, labels, context.loss_function, create_graph=False)
    parameter_size = sum(parameter.numel() for parameter in context.model.parameters())
    elements = size * parameter_size
    diagnostic = {
        "solver_method": settings["solver_method"],
        "rank_policy": "smallest_nonzero",
        "selection": "minimum_positive_numerical_singular_value",
        "relative_rank_tolerance": settings["relative_rank_tolerance"],
        "relative_tolerance": tolerance,
        "input_size": size, "gradient_size": parameter_size,
        "dense_elements": elements,
        "dense_max_elements": settings["dense_max_elements"],
        "dense_max_input_size": settings["dense_max_input_size"],
        "spectral_norm_is_certified_upper_bound": False,
        "global_minimum_certified": False,
        "lazy_initialized_for_operator": lazy_initialized,
        "stopping_criterion": "both_relative_singular_residuals_and_unit_norms",
    }
    if (settings["dense_max_elements"] and settings["dense_max_input_size"]
            and elements <= settings["dense_max_elements"]
            and size <= settings["dense_max_input_size"]):
        return _dense(context, images, labels, settings, tolerance, diagnostic, parameter_size)
    diagnostic.update(
        method="matrix_free_lanczos_positive", rank_complete=False,
        dense_skip_reason=("dense_disabled" if not settings["dense_max_elements"]
                           or not settings["dense_max_input_size"] else "dense_resource_cap"),
        left_vector_source="checked_live_jvp",
    )
    sigma, image, right, scale = minimum_krylov(
        context, images, labels, tolerance=tolerance,
        max_iterations=settings["krylov_max_iterations"],
        check_every=settings["krylov_check_every"],
        max_elements=settings["krylov_max_elements"], seed=settings["solver_seed"],
        diagnostic=diagnostic, rank_policy="smallest_nonzero",
        relative_rank_tolerance=settings["relative_rank_tolerance"],
    )
    if not torch.isfinite(sigma) or float(sigma) <= 0:
        _record(context, diagnostic, "numerical_nullspace",
                "No resolved positive singular direction was found by the bounded solver")
    left = image / _singular_norm(image)
    return _validate(context, sigma, left, right, scale, tolerance, diagnostic)
