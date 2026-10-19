import math

import torch
import torch.nn as nn

from transformations.transform_def import (
    grad_package,
    transformation_context,
)


# ============================================================
# basic helpers
# ============================================================

def flatten_gradients(
    package: grad_package
) -> torch.Tensor:
    """
    Flatten all parameter gradients into one vector.
    """

    return torch.cat([
        grad.reshape(-1)
        for grad in package.grads
    ])


def reshape_scores(
    flat_scores: torch.Tensor,
    package: grad_package
) -> list[torch.Tensor]:
    """
    Reshape a flat score vector back to gradient tensor shapes.
    """

    scores = []
    position = 0

    for grad in package.grads:

        size = grad.numel()

        score = flat_scores[
            position:position + size
        ].reshape(
            grad.shape
        )

        scores.append(
            score
        )

        position += size

    return scores


def gradient_size(
    package: grad_package
) -> int:
    """
    Return the total number of gradient values.
    """

    return sum(
        grad.numel()
        for grad in package.grads
    )


def batch_tensors(
    context: transformation_context
):
    """
    Convert context.batch into image and label tensors.
    """

    model = context.model

    device = next(
        model.parameters()
    ).device

    images = []
    labels = []

    for image, label in context.batch:

        images.append(
            image
        )

        labels.append(
            int(label)
        )

    if not images:
        raise ValueError(
            "Transformation batch cannot be empty"
        )

    images = torch.stack(
        images
    ).to(
        device
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return images, labels


def input_size(
    context: transformation_context
) -> int:
    """
    Return the total number of input values in the batch.
    """

    images, _ = batch_tensors(
        context
    )

    return images.numel()


# ============================================================
# loss / gradient helper
# ============================================================

def model_gradient_vector(
    model,
    images,
    labels,
    loss_function=None,
    create_graph=True,
):
    """
    Return the model parameter gradients as one flat vector.
    """

    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    model.zero_grad(
        set_to_none=True
    )

    predictions = model(
        images
    )

    loss = loss_function(
        predictions,
        labels,
    )

    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=create_graph,
    )

    return torch.cat([
        gradient.reshape(-1)
        for gradient in gradients
    ])


# ============================================================
# exact Jacobian
#
# only use this for small validation problems
# ============================================================

def gradient_jacobian(
    context: transformation_context
) -> torch.Tensor:
    """
    Construct the full Jacobian of model gradients
    with respect to the input.

    Shape:
        [gradient values, input values]

    This is extremely expensive and should only be
    used for small validation cases.
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    images = (
        images
        .detach()
        .clone()
    )

    images.requires_grad_(
        True
    )

    flat_gradients = model_gradient_vector(
        model=model,
        images=images,
        labels=labels,
        loss_function=context.loss_function,
        create_graph=True,
    )

    rows = []

    for gradient_value in flat_gradients:

        input_gradient = None
        if gradient_value.requires_grad:
            input_gradient = torch.autograd.grad(
                gradient_value, images, retain_graph=True, allow_unused=True,
            )[0]
        if input_gradient is None:
            input_gradient = torch.zeros_like(images)

        rows.append(
            input_gradient.reshape(-1)
        )

    return torch.stack(
        rows
    )


def exact_jtj_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J^T J)v using the exact Jacobian.

    Validation only.
    """

    jacobian = gradient_jacobian(
        context
    )

    vector = vector.to(
        jacobian.device
    )

    return (
        jacobian.T
        @ (
            jacobian
            @ vector
        )
    )


def exact_jjt_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J J^T)v using the exact Jacobian.

    Validation only.
    """

    jacobian = gradient_jacobian(
        context
    )

    vector = vector.to(
        jacobian.device
    )

    return (
        jacobian
        @ (
            jacobian.T
            @ vector
        )
    )


# ============================================================
# matrix-free Jacobian products
# ============================================================

def jacobian_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute Jv without explicitly constructing J.

    Input:
        vector in input space

    Output:
        vector in gradient space
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    vector = (
        vector
        .to(
            device=images.device,
            dtype=images.dtype,
        )
        .reshape_as(
            images
        )
    )

    def gradient_function(
        input_images
    ):

        return model_gradient_vector(
            model=model,
            images=input_images,
            labels=labels,
            loss_function=context.loss_function,
            create_graph=True,
        )

    _, jv = torch.autograd.functional.jvp(
        gradient_function,
        images,
        vector,
        create_graph=False,
        strict=False,
    )

    return jv


def jacobian_transpose_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute J^T v without explicitly constructing J.

    Input:
        vector in gradient space

    Output:
        flattened vector in input space
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    images = (
        images
        .detach()
        .clone()
    )

    images.requires_grad_(
        True
    )

    flat_gradients = model_gradient_vector(
        model=model,
        images=images,
        labels=labels,
        loss_function=context.loss_function,
        create_graph=True,
    )

    vector = (
        vector
        .to(
            device=flat_gradients.device,
            dtype=flat_gradients.dtype,
        )
        .reshape_as(
            flat_gradients
        )
    )

    if not flat_gradients.requires_grad:
        return torch.zeros_like(images).reshape(-1)

    jtv = torch.autograd.grad(
        outputs=flat_gradients,
        inputs=images,
        grad_outputs=vector,
        create_graph=False,
        allow_unused=True,
    )[0]

    if jtv is None:
        jtv = torch.zeros_like(images)
    return jtv.reshape(-1)


def jtj_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J^T J)v without constructing J.
    """

    jv = jacobian_vector_product(
        vector=vector,
        context=context,
    )

    return jacobian_transpose_vector_product(
        vector=jv,
        context=context,
    )


def jjt_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J J^T)v without constructing J.
    """

    jtv = jacobian_transpose_vector_product(
        vector=vector,
        context=context,
    )

    return jacobian_vector_product(
        vector=jtv,
        context=context,
    )


# ============================================================
# estimated Jacobian magnitude
# ============================================================

def estimated_jacobian_magnitude(
    context: transformation_context,
    samples: int = 10,
) -> torch.Tensor:
    """
    Estimate the L2 norm of every Jacobian row using
    randomized Rademacher probes.

    Returns one score for every gradient coordinate.
    """

    if samples <= 0:
        raise ValueError(
            "samples must be greater than zero"
        )

    images, _ = batch_tensors(
        context
    )

    estimate = None

    for _ in range(samples):

        vector = torch.empty_like(
            images
        )

        vector.bernoulli_(
            0.5
        )

        vector = (
            vector * 2
            - 1
        )

        jv = jacobian_vector_product(
            vector=vector,
            context=context,
        )

        squared = (
            jv ** 2
        )

        if estimate is None:
            estimate = squared
        else:
            estimate = (
                estimate
                + squared
            )

    estimate = (
        estimate
        / samples
    )

    return torch.sqrt(
        estimate
    )


# ============================================================
# conjugate gradient
# ============================================================

def conjugate_gradient(
    matrix_vector_func,
    b: torch.Tensor,
    iterations: int = 8,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Approximately solve a positive-definite system with relative tolerance.

    A fixed absolute cutoff on p.T A p incorrectly rejects small-scale systems.
    The caller remains responsible for checking the resulting solution residual.
    """
    if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("CG iterations and relative tolerance must be positive")
    x = torch.zeros_like(b)
    r = b - matrix_vector_func(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    threshold = tolerance * torch.linalg.vector_norm(b)
    if torch.linalg.vector_norm(r) <= threshold:
        return x
    for _ in range(iterations):
        ap = matrix_vector_func(p)
        denominator = torch.dot(p, ap)
        if not torch.isfinite(denominator):
            raise RuntimeError("Nonfinite conjugate-gradient denominator")
        if denominator <= 0:
            raise RuntimeError("CG requires a positive-definite operator")
        alpha = rs_old / denominator
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        if torch.sqrt(rs_new) <= threshold:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def record_numerical_diagnostic(context, name, values):
    """Keep JSON-compatible numerical status for the experiment metadata."""
    if getattr(context, "numerical_diagnostics", None) is None:
        context.numerical_diagnostics = {}
    context.numerical_diagnostics[name] = values


def _singular_settings(iterations, tolerance, exact_max_elements, exact_max_input_size):
    if iterations < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Singular iterations and relative tolerance must be positive")
    if exact_max_elements < 0 or exact_max_input_size < 0:
        raise ValueError("Exact Jacobian resource caps must be nonnegative")


def _small_jacobian(context, images, exact_max_elements, exact_max_input_size):
    # Build columns, rather than one reverse pass per parameter coordinate.
    # These two caps bound both storage/SVD work and derivative evaluations.
    size = images.numel()
    if size > exact_max_input_size or exact_max_elements == 0:
        return None
    if any(isinstance(p, torch.nn.parameter.UninitializedParameter)
           for p in context.model.parameters()):
        return None
    parameter_size = sum(p.numel() for p in context.model.parameters())
    if size * parameter_size > exact_max_elements:
        return None
    columns = []
    for index in range(size):
        direction = torch.zeros(size, dtype=images.dtype, device=images.device)
        direction[index] = 1
        columns.append(jacobian_vector_product(direction, context))
    matrix = torch.stack(columns, dim=1)
    if not torch.isfinite(matrix).all():
        raise RuntimeError("Nonfinite gradient Jacobian")
    return matrix


def _exact_singular_triplet(context, matrix, which, rank_policy):
    # Float64 SVD avoids losing small singular values solely to the decomposition.
    # Rank tolerance still reflects the precision of the computed Jacobian.
    left, values, right_h = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    relative_rank_tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps
    cutoff = float(values[0]) * relative_rank_tolerance
    rank = int((values > cutoff).sum())
    diagnostic = {
        "method": "exact_svd", "status": "converged",
        "input_size": matrix.shape[1], "gradient_size": matrix.shape[0],
        "rank": rank, "rank_policy": rank_policy,
        "relative_rank_tolerance": relative_rank_tolerance,
        "relative_residual": 0.0,
        "spectral_norm": float(values[0]),
        "spectral_norm_is_certified_upper_bound": False,
    }
    if which == "min" and (rank == 0 or (rank < matrix.shape[1] and rank_policy == "error")):
        diagnostic["status"] = "numerical_nullspace"
        record_numerical_diagnostic(context, "min_svd", diagnostic)
        raise RuntimeError(
            "Minimum singular direction has a numerical input nullspace; "
            "a zero singular value has no uniquely coupled left direction. "
            "For an explicit minimum-positive-singular-value experiment, use "
            "rank_policy='smallest_nonzero' (only when rank is nonzero)."
        )
    index = 0 if which == "max" else rank - 1
    sigma = values[index].to(matrix.dtype)
    u = left[:, index].to(matrix.dtype)
    v = right_h[index].to(matrix.dtype)
    if which == "min" and rank < matrix.shape[1]:
        diagnostic["status"] = "minimum_nonzero_by_explicit_policy"
    diagnostic["sigma"] = float(sigma)
    if which == "max" and rank == 0:
        u = torch.zeros_like(u)
        diagnostic["status"] = "zero_jacobian"
    if float(values[0]) > 0:
        left_residual = torch.linalg.vector_norm(matrix.T @ u - sigma * v)
        right_residual = torch.linalg.vector_norm(matrix @ v - sigma * u)
        diagnostic["relative_residual"] = float(
            torch.maximum(left_residual, right_residual) / values[0]
        )
    record_numerical_diagnostic(context, which + "_svd", diagnostic)
    return sigma, u, v


def _singular_norm(vector):
    # Gradient-space vectors can have millions of float32 coordinates. Their
    # float32 norm reduction can lose more than the requested SVD tolerance,
    # even when every JVP/VJP is accurate. Promote the reduction, not the model.
    return torch.linalg.vector_norm(vector, dtype=torch.float64)


def _random_unit(images):
    v = torch.randn(images.numel(), device=images.device, dtype=images.dtype)
    norm = _singular_norm(v)
    if not torch.isfinite(norm) or norm == 0:
        raise RuntimeError("Could not initialize singular direction")
    return v / norm


def _power_direction(context, images, iterations, tolerance=None, check_after=1):
    v = _random_unit(images)
    for step in range(iterations):
        image = jacobian_vector_product(v, context)
        sigma = _singular_norm(image)
        if not torch.isfinite(sigma):
            raise RuntimeError("Nonfinite singular power iteration")
        if sigma == 0:
            return sigma, image, v, step + 1
        # Normalizing between the products is algebraically the same power
        # direction as J.T(Jv), avoids squaring the scale, and lets early stop
        # use exactly the singular residual checked on the returned triplet.
        candidate = jacobian_transpose_vector_product(image / sigma, context)
        norm = _singular_norm(candidate)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite singular power iteration")
        if tolerance is not None and step + 1 >= check_after:
            relative = _singular_norm(candidate - sigma * v) / sigma
            if relative <= tolerance:
                return sigma, image, v, step + 1
        if norm == 0:
            break
        v = candidate / norm
    image = jacobian_vector_product(v, context)
    return _singular_norm(image), image, v, step + 1


def _checked_singular_triplet(context, sigma, image, v, scale, which, tolerance, diagnostic):
    # Do not amplify numerical leakage from Jv near an input-null direction.
    numerical_floor = math.sqrt(torch.finfo(image.dtype).eps) * float(scale)
    if not torch.isfinite(sigma) or float(sigma) <= numerical_floor:
        diagnostic.update(status="numerical_nullspace", sigma=float(sigma))
        record_numerical_diagnostic(context, which + "_svd", diagnostic)
        raise RuntimeError(
            "Singular direction is numerically null or unresolved; refusing to "
            "normalize Jacobian leakage. Use a bounded exact SVD on a smaller "
            "problem or an explicit rank-aware solver."
        )
    u = image / sigma
    residual = _singular_norm(jacobian_transpose_vector_product(u, context) - sigma * v)
    relative = float(residual / scale)
    diagnostic.update(sigma=float(sigma), relative_residual=relative)
    if not math.isfinite(relative) or relative > tolerance:
        diagnostic["status"] = "unconverged"
        record_numerical_diagnostic(context, which + "_svd", diagnostic)
        raise RuntimeError(
            f"{which}-SVD direction did not converge: relative singular residual "
            f"{relative:.3g} exceeds {tolerance:.3g}. Increase iterations/solve_iterations "
            "or use an exact SVD within the resource caps."
        )
    diagnostic["status"] = "converged_estimate"
    record_numerical_diagnostic(context, which + "_svd", diagnostic)
    return sigma, u, v


def max_singular_direction(
    context: transformation_context,
    iterations: int | None = None,
    tolerance: float = 1e-4,
    exact_max_elements: int = 200000,
    exact_max_input_size: int = 64,
    max_iterations: int | None = None,
):
    """Return a checked dominant triplet, with a bounded exact-SVD fallback.

    Matrix-free convergence is checked by the singular residual; it is not a
    certified upper bound on the spectral norm or proof of global extremality.
    Set either exact resource cap to zero to force matrix-free operation.
    With omitted iterations, refine from 10 to at most 500 power steps, stopping
    on a checked residual. An explicit iterations value is a hard limit unless
    max_iterations explicitly grants a larger adaptive budget.
    """
    initial_iterations = 10 if iterations is None else iterations
    cap = (500 if iterations is None else iterations) if max_iterations is None else max_iterations
    _singular_settings(initial_iterations, tolerance, exact_max_elements, exact_max_input_size)
    if cap < initial_iterations:
        raise ValueError("max_iterations must be at least the initial iterations")
    images, _ = batch_tensors(context)
    matrix = _small_jacobian(context, images, exact_max_elements, exact_max_input_size)
    if matrix is not None:
        return _exact_singular_triplet(context, matrix, "max", "maximum")
    sigma, image, v, used_iterations = _power_direction(
        context, images, cap, tolerance=tolerance, check_after=initial_iterations,
    )
    diagnostic = {
        "method": "matrix_free_power", "iterations": used_iterations,
        "initial_iterations": initial_iterations, "max_iterations": cap,
        "relative_tolerance": tolerance,
        "stopping_criterion": "relative_singular_residual",
        "norm_accumulation_dtype": "float64",
        "spectral_norm_is_certified_upper_bound": False,
    }
    if float(sigma) == 0:
        diagnostic.update(status="zero_observed_product", sigma=0.0, relative_residual=0.0)
        record_numerical_diagnostic(context, "max_svd", diagnostic)
        return sigma, torch.zeros_like(image), v
    return _checked_singular_triplet(context, sigma, image, v, sigma, "max", tolerance, diagnostic)


def min_singular_direction(
    context: transformation_context,
    iterations: int = 5,
    solve_iterations: int = 8,
    damping: float = 1e-4,
    tolerance: float = 1e-4,
    exact_max_elements: int = 200000,
    exact_max_input_size: int = 64,
    rank_policy: str = "error",
):
    """Return a checked minimum triplet, rejecting unresolved/null directions.

    ``damping`` is relative to the estimated squared spectral norm, preserving
    directions under scalar changes of the gradient map. Exact SVD is used only
    within both resource caps. The default rejects input-null singular values;
    ``smallest_nonzero`` explicitly chooses the least positive triplet when an
    exact SVD identifies rank. Matrix-free output remains an estimate.
    """
    _singular_settings(iterations, tolerance, exact_max_elements, exact_max_input_size)
    if not math.isfinite(damping) or damping <= 0 or solve_iterations < 1:
        raise ValueError("Damping and solve iterations must be positive")
    if rank_policy not in {"error", "smallest_nonzero"}:
        raise ValueError("rank_policy must be 'error' or 'smallest_nonzero'")
    images, _ = batch_tensors(context)
    matrix = _small_jacobian(context, images, exact_max_elements, exact_max_input_size)
    if matrix is not None:
        return _exact_singular_triplet(context, matrix, "min", rank_policy)
    if rank_policy != "error":
        raise RuntimeError("smallest_nonzero rank policy requires exact SVD within resource caps")
    scale, _, _, _ = _power_direction(context, images, max(5, iterations))
    if not torch.isfinite(scale) or scale == 0:
        record_numerical_diagnostic(context, "min_svd", {"status": "zero_jacobian_observed", "method": "matrix_free_inverse"})
        raise RuntimeError("No nonzero Jacobian direction observed for minimum-SVD estimation")
    v = _random_unit(images)
    def normalized_operator(direction):
        # Dividing Jv first avoids underflow/overflow from squaring the scale.
        jv = jacobian_vector_product(direction, context) / scale
        return jacobian_transpose_vector_product(jv, context) / scale + damping * direction
    for _ in range(iterations):
        z = conjugate_gradient(normalized_operator, v, iterations=solve_iterations,
                               tolerance=min(tolerance * .1, 1e-6))
        norm = _singular_norm(z)
        if not torch.isfinite(norm) or norm == 0:
            raise RuntimeError("Minimum-SVD inverse solve produced an invalid direction")
        v = z / norm
    image = jacobian_vector_product(v, context)
    sigma = _singular_norm(image)
    diagnostic = {
        "method": "matrix_free_inverse", "iterations": iterations,
        "solve_iterations": solve_iterations, "relative_damping": damping,
        "relative_tolerance": tolerance, "rank_policy": rank_policy,
        "spectral_scale_estimate": float(scale),
        "norm_accumulation_dtype": "float64",
        "spectral_norm_is_certified_upper_bound": False,
    }
    return _checked_singular_triplet(context, sigma, image, v, scale, "min", tolerance, diagnostic)
