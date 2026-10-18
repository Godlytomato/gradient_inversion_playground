"""Independent lower-bound geometry and reference-scaling error contracts."""
import contextlib
import io
import math
import unittest
import warnings
from unittest.mock import patch

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.transformation_implimentations.threasholds.lower_bound_threashold import lower_bound_threshold
from transformations.transformation_implimentations.threasholds.max_svd_reference_scale import max_svd_reference_scale


BOUND_MODULE = "transformations.transformation_implimentations.threasholds.lower_bound_threashold"
REFERENCE_MODULE = "transformations.transformation_implimentations.threasholds.max_svd_reference_scale"


class AffineGradientMap(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(2, dtype=dtype))
        self.register_buffer("matrix", torch.diag(torch.tensor([2., 1.], dtype=dtype)))

    def forward(self, images):
        return ((images.flatten(1) @ self.matrix.T) * self.theta).sum(1)


class LowerBoundTests(unittest.TestCase):
    def setUp(self):
        self.context, self.package = self.geometry(torch.float64)
        self.mask = [torch.ones(2, dtype=torch.bool)]
        self.noise = [torch.tensor([1., 0.], dtype=torch.float64)]

    @staticmethod
    def geometry(dtype):
        model = AffineGradientMap(dtype)
        context = transformation_context(
            model, [(torch.ones(2, dtype=dtype), 0)],
            loss_function=lambda prediction, labels: prediction.sum(),
        )
        return context, grad_package([torch.tensor([2., 1.], dtype=dtype)])

    def test_finds_first_root_when_budget_endpoint_is_infeasible(self):
        # F(s)=2s-a-b*sqrt((s-2)^2+1) is concave. It is negative
        # at both endpoints but positive near its interior maximum s=3.
        length = math.sqrt(.02)
        a, b = length * (4 + 20 * length), 20 * length
        quadratic = (4-b*b, -4*a+4*b*b, a*a-5*b*b)
        aa, bb, cc = quadratic
        expected = min((-bb + sign * math.sqrt(bb*bb-4*aa*cc))/(2*aa)
                       for sign in (-1, 1))
        with contextlib.redirect_stdout(io.StringIO()):
            output = lower_bound_threshold(
                self.package, self.noise, self.mask, self.context,
                target_error=.01, mu_j=10., mu_j_method="manual", max_scale=10.,
            )
        actual = float(output[0].norm())
        self.assertAlmostEqual(actual, expected, places=12)
        diagnostic = self.context.numerical_diagnostics["lower_bound"]
        self.assertEqual(diagnostic["status"], "feasible_under_supplied_constants")
        self.assertAlmostEqual(diagnostic["peak_scale"], 3., places=6)
        self.assertLess(2*diagnostic["absolute_budget"] - a
                        - b*math.hypot(diagnostic["absolute_budget"]-2, 1), 0)

    def test_infeasible_bound_returns_finite_interior_best_effort(self):
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            output = lower_bound_threshold(
                self.package, self.noise, self.mask, self.context,
                target_error=.02, mu_j=10., mu_j_method="manual", max_scale=10.,
            )
        self.assertAlmostEqual(float(output[0].norm()), 2+1/math.sqrt(3), places=6)
        diagnostic = self.context.numerical_diagnostics["lower_bound"]
        self.assertEqual(diagnostic["status"], "infeasible_best_effort")
        self.assertLess(diagnostic["peak_residual"], 0)
        self.assertTrue(any("INFEASIBLE" in str(warning.message) for warning in caught))

    def test_bound_preserves_failed_spectral_stage(self):
        with patch(BOUND_MODULE+".max_singular_direction", side_effect=RuntimeError("not converged")):
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                lower_bound_threshold(self.package, self.noise, self.mask, self.context,
                                      target_error=.02, mu_j=.1, mu_j_method="manual")
        diagnostic = self.context.numerical_diagnostics["lower_bound"]
        self.assertEqual(diagnostic["status"], "numerical_error")
        self.assertEqual(diagnostic["failed_stage"], "estimating_spectral_norm")
        self.assertEqual(diagnostic["mu_j"], .1)

    def test_zero_target_reference_does_not_require_a_singular_solve(self):
        with patch(REFERENCE_MODULE+".max_svd_alignment", side_effect=AssertionError("unexpected solve")):
            output = max_svd_reference_scale(self.package, self.noise, self.mask,
                                             self.context, target_error=0.)
        self.assertEqual(float(output[0].norm()), 0.)
        self.assertEqual(self.context.numerical_diagnostics["max_svd_reference_scale"]["status"], "zero_target")

    def test_reference_preserves_failed_direction_stage(self):
        with patch(REFERENCE_MODULE+".max_svd_alignment", side_effect=RuntimeError("not converged")):
            with self.assertRaisesRegex(RuntimeError, "not converged"):
                max_svd_reference_scale(self.package, self.noise, self.mask,
                                        self.context, target_error=.02)
        diagnostic = self.context.numerical_diagnostics["max_svd_reference_scale"]
        self.assertEqual(diagnostic["status"], "reference_direction_error")
        self.assertEqual(diagnostic["message"], "not converged")

    def test_reference_rejects_invalid_directions_and_preserves_extreme_finite_norms(self):
        context, package = self.geometry(torch.float32)
        settings = dict(target_error=.02, mu_j=0., mu_j_method="manual", max_scale=1.)
        for magnitude in (1e-40, 1e30):
            with self.subTest(magnitude=magnitude), contextlib.redirect_stdout(io.StringIO()):
                output = max_svd_reference_scale(
                    package, [torch.tensor([0., magnitude])], self.mask, context, **settings,
                )
                self.assertTrue(torch.isfinite(output[0]).all())
                self.assertEqual(output[0].dtype, torch.float32)
                self.assertAlmostEqual(float(output[0].norm()), .4, places=6)
        with patch(REFERENCE_MODULE+".max_svd_alignment") as direction:
            with self.assertRaisesRegex(ValueError, "finite"):
                max_svd_reference_scale(package, [torch.tensor([float("nan"), 1.])],
                                        self.mask, context, **settings)
            direction.assert_not_called()

    def test_reference_rejects_output_that_overflows_the_noise_dtype(self):
        context, package = self.geometry(torch.float32)
        # Each reference coordinate is finite, but concentrating its norm onto
        # one coordinate exceeds float32's representable range.
        large_reference = [torch.full((2,), torch.finfo(torch.float32).max)]
        with patch(REFERENCE_MODULE+".max_svd_alignment", return_value=large_reference):
            with patch(REFERENCE_MODULE+".lower_bound_threshold", return_value=large_reference):
                with self.assertRaisesRegex(ValueError, "represented finitely"):
                    max_svd_reference_scale(package, [torch.tensor([1., 0.])],
                                            self.mask, context, target_error=.02)
        self.assertEqual(context.numerical_diagnostics["max_svd_reference_scale"]["status"],
                         "nonfinite_scaled_noise")


if __name__ == "__main__":
    unittest.main()
