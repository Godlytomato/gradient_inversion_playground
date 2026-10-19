"""Bounded input-space Rayleigh--Ritz search for a minimum singular mode.

Only input-space basis vectors/products are retained. No gradient Jacobian or
gradient-space Krylov basis is materialized. The result remains an estimate;
all accepted triplets are checked with the public autograd JVP/VJP operators.
"""
import math

import torch


def _derivative_products(context, images, labels, probe, diagnostic):
    from transformations.ranking_implimentations.transform_utils import (
        jacobian_vector_product, jacobian_transpose_vector_product,
    )

    def ordinary_jvp(vector):
        return jacobian_vector_product(vector, context)

    def ordinary_vjp(vector):
        return jacobian_transpose_vector_product(vector, context)

    # Forward-over-reverse differentiation avoids the expensive reverse-over-
    # reverse JVP used by the independent final validation. Both differentiate
    # the same fixed loss/model/input. Unsupported custom ops keep the ordinary
    # backend, explicitly recorded in the diagnostic.
    try:
        parameters = {name: value.detach() for name, value in context.model.named_parameters()}
        buffers = {name: value.detach() for name, value in context.model.named_buffers()}
        loss_function = context.loss_function or torch.nn.CrossEntropyLoss()

        def loss(parameters, inputs):
            predictions = torch.func.functional_call(context.model, (parameters, buffers), (inputs,))
            return loss_function(predictions, labels)

        differentiate = torch.func.grad(loss)

        def gradient_map(inputs):
            return torch.cat([value.reshape(-1) for value in differentiate(parameters, inputs).values()])

        _, transpose = torch.func.vjp(gradient_map, images)

        def forward_jvp(vector):
            direction = vector.to(images.dtype).reshape_as(images)
            _, image = torch.func.jvp(gradient_map, (images,), (direction,))
            return image.detach()

        def reverse_vjp(vector):
            return transpose(vector)[0].detach().reshape(-1)

        forward_jvp(probe)
    except (NotImplementedError, RuntimeError) as error:
        diagnostic.update(derivative_backend="autograd", backend_fallback_reason=str(error))
        return ordinary_jvp, ordinary_vjp
    diagnostic["derivative_backend"] = "torch_func_forward_over_reverse"
    return forward_jvp, reverse_vjp


def minimum_krylov(context, images, labels, *, tolerance, max_iterations,
                   check_every, max_elements, seed, diagnostic,
                   rank_policy="error", relative_rank_tolerance=None):
    from transformations.ranking_implimentations.transform_utils import (
        _singular_norm, jacobian_vector_product,
        jacobian_transpose_vector_product, record_numerical_diagnostic,
    )

    size = images.numel()
    positive = rank_policy == "smallest_nonzero"
    if positive and (relative_rank_tolerance is None
                     or not math.isfinite(relative_rank_tolerance)
                     or not 0 < relative_rank_tolerance < 1):
        raise ValueError("Minimum-positive search requires a relative rank tolerance in (0, 1)")
    cap = min(size, max_iterations)
    elements = 2 * size * cap
    diagnostic.update(input_size=size, max_iterations=max_iterations,
                      effective_max_iterations=cap, basis_elements=elements,
                      max_basis_elements=max_elements, solver_seed=seed,
                      global_minimum_certified=False, residual_history=[])
    if elements > max_elements:
        diagnostic["status"] = "resource_cap"
        record_numerical_diagnostic(context, "min_svd", diagnostic)
        raise RuntimeError(
            f"Minimum-SVD basis needs {elements} elements, exceeding the declared "
            f"krylov_max_elements={max_elements}. Reduce batch/input size or the "
            "iteration cap, or explicitly increase the memory cap."
        )
    generator = torch.Generator(device=images.device).manual_seed(seed)

    def random_unit():
        vector = torch.randn(size, dtype=images.dtype, device=images.device, generator=generator)
        return vector / _singular_norm(vector)

    probe = random_unit()
    jvp, vjp = _derivative_products(context, images, labels, probe, diagnostic)
    # A scale estimate only: it is never represented as a spectral upper bound.
    for _ in range(5):
        image = jvp(probe)
        sigma = _singular_norm(image)
        if not torch.isfinite(sigma) or sigma == 0:
            break
        candidate = vjp(image / sigma)
        length = _singular_norm(candidate)
        if not torch.isfinite(length) or length == 0:
            break
        probe = candidate / length
    scale = _singular_norm(jvp(probe))
    diagnostic.update(scale_iterations=5, spectral_scale_estimate=float(scale))
    if not torch.isfinite(scale) or scale == 0:
        diagnostic["status"] = "zero_jacobian_observed"
        record_numerical_diagnostic(context, "min_svd", diagnostic)
        raise RuntimeError("No nonzero Jacobian direction observed for minimum-SVD estimation")

    def operator(vector):
        return (vjp(jvp(vector) / scale) / scale).to(torch.float64)

    basis = torch.empty((size, cap), device=images.device, dtype=torch.float64)
    products = torch.empty_like(basis)
    q = random_unit().to(torch.float64)
    q = q / _singular_norm(q)
    last_triplet = None
    reference_scale = scale
    for step in range(cap):
        basis[:, step] = q
        product = operator(q)
        if not torch.isfinite(product).all():
            diagnostic.update(status="nonfinite_product", iterations=step + 1)
            record_numerical_diagnostic(context, "min_svd", diagnostic)
            raise RuntimeError("Nonfinite minimum-SVD Krylov product")
        products[:, step] = product
        active = basis[:, :step + 1]
        active_products = products[:, :step + 1]
        remainder = product - active @ (active.T @ product)
        remainder -= active @ (active.T @ remainder)
        length = _singular_norm(remainder)
        # An invariant subspace may exclude the minimum. Explore an independent
        # orthogonal direction at breakdown when the input space is not full.
        breakdown = float(length) <= 64 * torch.finfo(torch.float64).eps * max(1., float(_singular_norm(product)))
        unexplored_breakdown = breakdown and step + 1 < size
        should_check = (step + 1) % check_every == 0 or step + 1 == cap or breakdown
        if should_check:
            projected = active.T @ active_products
            values, rotations = torch.linalg.eigh((projected + projected.T) * .5)
            index = 0
            if positive:
                # B uses the fixed initial scale. A Ritz estimate of the top
                # mode improves the reported cutoff without changing B midway.
                reference_scale = scale * math.sqrt(max(1., float(values[-1])))
                cutoff = relative_rank_tolerance * float(reference_scale)
                diagnostic.update(rank_cutoff=cutoff,
                                  spectral_scale_estimate=float(reference_scale),
                                  normal_operator_scale=float(scale))
                if step + 1 == size:
                    cutoff_squared = (cutoff / float(scale)) ** 2
                    # Forming J.T J squares the condition number. Accumulating
                    # its projection in float64 cannot recover small modes
                    # already lost by the original-precision JVP/VJP products.
                    # A negative/weak Ritz value could hide a genuine positive
                    # singular mode; skipping it would return an interior mode
                    # with a perfectly good residual. Refuse rank selection in
                    # this unresolved band rather than clipping its spectrum.
                    normal_scale = max(1., abs(float(values[-1])))
                    precision_floor = max(
                        torch.finfo(images.dtype).eps * normal_scale,
                        float(_singular_norm(projected - projected.T)),
                        64 * torch.finfo(torch.float64).eps * size * normal_scale,
                    )
                    diagnostic.update(
                        normal_rank_cutoff_squared=cutoff_squared,
                        normal_operator_precision_floor=precision_floor,
                        operator_dtype=str(images.dtype).removeprefix("torch."),
                    )
                    if cutoff_squared <= precision_floor and float(values[0]) <= precision_floor:
                        diagnostic.update(status="rank_unresolved_at_operator_precision",
                                          iterations=step + 1,
                                          lowest_ritz_value=float(values[0]))
                        record_numerical_diagnostic(context, "min_svd", diagnostic)
                        raise RuntimeError(
                            "Minimum-positive rank cannot be resolved at the original operator "
                            "precision: the squared rank cutoff lies below the normal-operator "
                            "roundoff band. Use the bounded dense decomposition or explicitly "
                            "run a higher-precision model; no interior mode was substituted."
                        )
                    eligible = torch.nonzero(values > cutoff_squared).flatten()
                    if eligible.numel() == 0:
                        diagnostic.update(status="zero_jacobian_or_no_positive_mode", iterations=step + 1)
                        record_numerical_diagnostic(context, "min_svd", diagnostic)
                        raise RuntimeError("No positive singular mode exceeds the declared rank cutoff")
                    index = int(eligible[0])
                    diagnostic.update(rank_complete=True, numerical_rank=int(eligible.numel()),
                                      rank=int(eligible.numel()), input_nullity=size - int(eligible.numel()),
                                      rank_resolution="complete_input_projection")
                else:
                    # A partial projected spectrum cannot establish how many
                    # null/weak modes precede its first retained positive Ritz
                    # value. Keep searching the bottom mode; never skip it to
                    # accept an arbitrary interior mode merely above cutoff.
                    diagnostic.update(rank_complete=False,
                                      rank_resolution="lowest_mode_estimate_above_cutoff")
            vector = (active @ rotations[:, index]).to(images.dtype)
            vector = vector / _singular_norm(vector)
            # Always check the original operator, independently of the faster
            # derivative backend and normal-equations Ritz residual.
            image = jacobian_vector_product(vector, context)
            sigma = _singular_norm(image)
            if sigma > 0 and torch.isfinite(sigma):
                relative = float(_singular_norm(
                    jacobian_transpose_vector_product(image / sigma, context) - sigma * vector,
                ) / reference_scale)
            else:
                relative = math.inf
            reported_residual = relative if math.isfinite(relative) else None
            projected_norm = max(float(_singular_norm(projected)), torch.finfo(torch.float64).tiny)
            diagnostic.update(iterations=step + 1, lowest_ritz_value=float(values[0]),
                              projected_relative_skew=float(_singular_norm(projected - projected.T)) / projected_norm,
                              sigma=float(sigma), relative_residual=reported_residual)
            diagnostic["residual_history"].append({
                "iterations": step + 1, "sigma": float(sigma), "relative_residual": reported_residual,
            })
            last_triplet = sigma, image, vector
            above_cutoff = not positive or float(sigma) > cutoff
            if positive and step + 1 == size and not above_cutoff:
                diagnostic["status"] = "rank_unresolved"
                record_numerical_diagnostic(context, "min_svd", diagnostic)
                raise RuntimeError(
                    "A projected positive Ritz mode fails the original-operator rank cutoff; "
                    "rank is unresolved at this precision. Use the bounded dense decomposition."
                )
            if not unexplored_breakdown and above_cutoff and relative <= tolerance:
                return (*last_triplet, reference_scale)
        if step + 1 == cap:
            break
        if unexplored_breakdown:
            # Random augmentation is local to the solver's independent stream.
            remainder = random_unit().to(torch.float64)
            remainder -= active @ (active.T @ remainder)
            remainder -= active @ (active.T @ remainder)
            length = _singular_norm(remainder)
            if length <= 64 * torch.finfo(torch.float64).eps:
                diagnostic["status"] = "unresolved_subspace"
                record_numerical_diagnostic(context, "min_svd", diagnostic)
                raise RuntimeError("Minimum-SVD could not expand an unresolved input subspace")
        q = remainder / length
    if unexplored_breakdown:
        diagnostic["status"] = "unresolved_subspace"
        record_numerical_diagnostic(context, "min_svd", diagnostic)
        raise RuntimeError(
            "Minimum-SVD exhausted its cap in an invariant subspace while "
            "unsearched input directions remain; refusing an interior mode."
        )
    if positive and float(last_triplet[0]) <= relative_rank_tolerance * float(reference_scale):
        diagnostic["status"] = "positive_rank_unresolved_within_cap"
        record_numerical_diagnostic(context, "min_svd", diagnostic)
        raise RuntimeError(
            "The bounded search reached null/weak modes but cannot resolve the first "
            "positive mode from a partial input subspace. Increase the declared dense "
            "caps or permit a complete input-space search; no interior mode was substituted."
        )
    return (*last_triplet, reference_scale)
