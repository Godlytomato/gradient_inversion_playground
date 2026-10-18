"""Reproduce applying the conditional bound outside its exact-fit assumption.

Uses exact float64 derivatives of a smooth cross-entropy model, one input in
[0,1], and analytically known sigma and mu_J. It does not access the database.
The supplied proofs.docx assumes zero gradient-matching residual. This example
violates that assumption; it does not refute the conditional theorem. It shows
why a feasible calibration alone cannot establish that theorem's applicability.
See docs/LOWER_BOUND_AUDIT.md.
"""
import contextlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.ranking_implimentations.transform_utils import (
    gradient_jacobian, model_gradient_vector,
)
from transformations.transformation_implimentations.threasholds.lower_bound_threashold import lower_bound_threshold


class CrossEntropyGradientMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, images):
        displacement = images.flatten(1).sum(1) - .5
        g = 10. + .1 * displacement + .5 * displacement.square()
        signal = 2. * self.theta * g
        return torch.stack((signal, torch.zeros_like(signal)), dim=1)


def reproduce():
    model = CrossEntropyGradientMap()
    images = torch.tensor([[.5]], dtype=torch.float64)
    labels = torch.tensor([1])
    context = transformation_context(model, [(images[0], 1)])
    gradient = model_gradient_vector(model, images, labels, create_graph=False)
    jacobian = gradient_jacobian(context)
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        noise = lower_bound_threshold(
            grad_package([gradient.squeeze()]),
            [torch.tensor(1., dtype=torch.float64)],
            [torch.tensor(True)], context,
            target_error=.02, mu_j=1., mu_j_method="manual",
            bound_method="legacy_proof_v1",
        )[0]
    observed = float(gradient.squeeze() + noise)
    # g(x) = 9.995 + 0.5*(x-0.4)^2. If y < min(g), squared
    # gradient distance is minimized uniquely at x=0.4 inside [0,1].
    if observed >= 9.995:
        raise RuntimeError("Calibration changed; this analytic optimum needs rechecking")
    optimum = torch.tensor([[.4]], dtype=torch.float64)
    optimum_gradient = model_gradient_vector(model, optimum, labels, create_graph=False)
    actual_mse = float((images - optimum).square().mean())
    status = context.numerical_diagnostics["lower_bound"]["status"]
    return {
        "model": "logits=[2*theta*g(x),0], theta=0, label=1, CrossEntropyLoss",
        "gradient_map": "g(x)=9.995+0.5*(x-0.4)^2",
        "input_domain": [0., 1.],
        "original_input": .5,
        "original_gradient": float(gradient.squeeze()),
        "exact_jacobian": jacobian.tolist(),
        "exact_sigma": .1, "exact_mu_j": 1.,
        "returned_noise": float(noise), "observed_gradient": observed,
        "global_optimum_input": .4,
        "global_optimum_gradient": float(optimum_gradient.squeeze()),
        "minimum_gradient_matching_residual": float(abs(optimum_gradient.squeeze() - observed)),
        "proof_exact_fit_assumption_satisfied": False,
        "global_optimum_mse": actual_mse, "requested_target_mse": .02,
        "reported_feasible_but_target_violated": (
            status == "feasible_under_supplied_constants" and actual_mse < .02
        ),
        "diagnostics": context.numerical_diagnostics,
        "calibration_output": console.getvalue(),
    }


if __name__ == "__main__":
    print(json.dumps(reproduce(), indent=2, allow_nan=False))
