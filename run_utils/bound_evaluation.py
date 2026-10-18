"""Evaluate conditional bounds using actual saved reconstruction residuals.

This is evaluation-only: original images/labels and gradients are never sent to
the attack. Reference methods are evaluated in their ACTUAL noise direction.
"""
import math

import torch

from observer.batch_matching import align_batch
from transformations.transform_def import transformation_context
from transformations.reconstruction_bound import residual_aware_bound
from transformations.ranking_implimentations.transform_utils import (
    batch_tensors, model_gradient_vector, jacobian_transpose_vector_product,
)


def evaluate_reconstruction_bounds(context, images, batch_matching="assignment"):
    metadata = context.experiment_metadata
    calibration = metadata.get("numerical_diagnostics", {}).get("lower_bound", {})
    if calibration.get("bound_method") != "directional_residual_v1":
        return {}
    try:
        original = context.original_gradients
        if original is None:
            raise ValueError("Original private gradients unavailable for bound evaluation")
        transformation = transformation_context(
            context.model, context.batch,
            loss_function=getattr(context, "loss_function", None),
        )
        truth, labels = batch_tensors(transformation)
        released = torch.cat([g.detach().reshape(-1) for g in context.gradients.grads])
        baseline = torch.cat([g.detach().reshape(-1) for g in original])
        delta = released - baseline
        amplitude = float(torch.linalg.vector_norm(delta.to(torch.float64)))
        if amplitude:
            direction = (delta.to(torch.float64) / amplitude).to(delta.dtype)
            gain = float(torch.linalg.vector_norm(
                jacobian_transpose_vector_product(direction, transformation).to(torch.float64)))
        else:
            gain = 0.0
        mu = calibration.get("mu_j", 0.) if amplitude == 0 else calibration["mu_j"]
        residual_budget = calibration["residual_l2"] + calibration["residual_fraction"] * amplitude
        output, cache = {}, []
        for name, reconstruction in images.items():
            if reconstruction is None:
                output[name] = {"status": "unavailable", "reason": "No reconstruction saved", "certified": False}
                continue
            cached = next((value for image, value in cache if torch.equal(image, reconstruction)), None)
            if cached is not None:
                output[name] = dict(cached)
                continue
            try:
                aligned, assignment = align_batch(reconstruction, truth, batch_matching)
                evaluated = model_gradient_vector(
                    context.model, aligned.detach().to(truth), labels,
                    loss_function=transformation.loss_function, create_graph=False,
                ).detach()
                epsilon = float(torch.linalg.vector_norm((evaluated - released).to(torch.float64)))
                result = residual_aware_bound(amplitude, epsilon, gain, mu)
                lower = result["lower_bound_l2"]
                candidate = None if lower is None else lower * lower / truth.numel()
                observed = float((aligned.detach().to(torch.float64) - truth.to(torch.float64)).square().mean())
                error_l2 = math.sqrt(observed * truth.numel())
                observed_mu = (max(0., 2*(amplitude-epsilon-gain*error_l2)/(error_l2**2))
                               if error_l2 else 0.)
                # A contradiction is evidence against the assumed constants or
                # smoothness, and must not be published as a valid lower bound.
                tolerance = 32 * torch.finfo(truth.dtype).eps * max(observed, candidate or 0., 1e-30)
                contradicted = candidate is not None and candidate > observed + tolerance
                value = {
                    "status": "assumptions_contradicted" if contradicted else result["status"],
                    "bound_method": "directional_residual_v1", "certified": False,
                    "uniform_smoothness_verified": False,
                    "labels": "true_labels_for_evaluation", "assignment": assignment,
                    "noise_l2": amplitude, "gradient_residual_l2": epsilon,
                    "direction_effect": gain, "mu_j": mu,
                    "observed_taylor_mu_lower_estimate": observed_mu,
                    "nonsmooth_module_types": calibration.get("nonsmooth_module_types", []),
                    "calibration_residual_budget_l2": residual_budget,
                    "calibration_residual_condition_met": epsilon <= residual_budget,
                    "candidate_lower_bound_mse": candidate,
                    "conditional_lower_bound_l2": None if contradicted else lower,
                    "conditional_lower_bound_mse": None if contradicted else candidate,
                    "observed_mse": observed,
                }
                cache.append((reconstruction, value))
                output[name] = value
            except Exception as error:
                output[name] = {"status": "unavailable", "reason": f"{type(error).__name__}: {error}", "certified": False}
        return output
    except Exception as error:
        # Evaluation diagnostics must not discard a completed attack/result.
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}", "certified": False}
