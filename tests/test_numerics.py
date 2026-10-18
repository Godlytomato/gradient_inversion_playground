"""Numerical contracts against explicitly prescribed tiny gradient Jacobians."""
import contextlib
import io
import json
import math
import unittest
import warnings

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.ranking_implimentations.transform_utils import (
    conjugate_gradient, gradient_jacobian, jacobian_vector_product,
    jacobian_transpose_vector_product, jtj_vector_product, jjt_vector_product,
    min_singular_direction, max_singular_direction,
)
from transformations.ranking_implimentations.min_svd import min_svd
from transformations.ranking_implimentations.exact_i2f import exact_i2f
from transformations.transformation_implimentations.min_svd_alignment import min_svd_alignment
from transformations.transformation_implimentations.threasholds.lower_bound_threashold import lower_bound_threshold
from transformations.transformation_implimentations.threasholds.max_svd_reference_scale import max_svd_reference_scale


class PrescribedGradientMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(matrix.shape[0], dtype=matrix.dtype))
        self.register_buffer("matrix", matrix)

    def forward(self, images):
        return ((images.flatten(1) @ self.matrix.T) * self.theta).sum(1)


def prescribed(matrix):
    matrix = matrix.to(torch.float64)
    image = torch.ones(matrix.shape[1], dtype=matrix.dtype)
    context = transformation_context(
        PrescribedGradientMap(matrix), [(image, 0)],
        loss_function=lambda prediction, labels: prediction.sum(),
    )
    return context, grad_package([matrix @ image])


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_products_match_prescribed_jacobian(self):
        matrix = torch.tensor([[1.7, .2, -.1], [0., .6, .3], [.2, -.1, .2], [1., -.2, .3]], dtype=torch.float64)
        context, _ = prescribed(matrix)
        v = torch.tensor([.3, -.6, .7], dtype=torch.float64)
        w = torch.tensor([.1, -.2, .5, .3], dtype=torch.float64)
        torch.testing.assert_close(gradient_jacobian(context), matrix)
        torch.testing.assert_close(jacobian_vector_product(v, context), matrix @ v)
        torch.testing.assert_close(jacobian_transpose_vector_product(w, context), matrix.T @ w)
        torch.testing.assert_close(jtj_vector_product(v, context), matrix.T @ matrix @ v)
        torch.testing.assert_close(jjt_vector_product(w, context), matrix @ matrix.T @ w)

    def test_minimum_direction_is_invariant_under_rescaling(self):
        for scale in (1e-8, .001, 1., 1000.):
            with self.subTest(scale=scale):
                matrix = torch.diag(torch.tensor([1., .001], dtype=torch.float64)) * scale
                context, _ = prescribed(matrix)
                sigma, u, v = min_singular_direction(context)
                self.assertAlmostEqual(float(sigma / scale), .001, places=12)
                self.assertGreater(float(abs(u[1])), .999999)
                self.assertLess(float((matrix.T @ u - sigma * v).norm() / scale), 1e-12)

    def test_rank_deficiency_is_explicitly_rejected(self):
        context, _ = prescribed(torch.diag(torch.tensor([1., 0.])))
        with self.assertRaisesRegex(RuntimeError, "nullspace"):
            min_singular_direction(context)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["status"], "numerical_nullspace")
        sigma, _, _ = min_singular_direction(context, rank_policy="smallest_nonzero")
        self.assertEqual(float(sigma), 1.)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["status"], "minimum_nonzero_by_explicit_policy")

    def test_cross_entropy_input_nullspace_is_rejected(self):
        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(1, 2).double()
            def forward(self, x):
                return self.fc(x.flatten(1)[:, :1])
        context = transformation_context(Classifier(), [(torch.tensor([.2, .4], dtype=torch.float64), 1)])
        with self.assertRaisesRegex(RuntimeError, "nullspace"):
            min_singular_direction(context)

    def test_zero_rank_has_no_nonzero_minimum(self):
        context, _ = prescribed(torch.zeros(2, 2))
        with self.assertRaisesRegex(RuntimeError, "nullspace"):
            min_singular_direction(context, rank_policy="smallest_nonzero")
        sigma, u, _ = max_singular_direction(context)
        self.assertEqual(float(sigma), 0.)
        self.assertEqual(float(u.norm()), 0.)

    def test_matrix_free_triplets_are_checked_and_scale_aware(self):
        for scale in (.001, 1., 1000.):
            with self.subTest(scale=scale):
                matrix = torch.diag(torch.tensor([2., 1.], dtype=torch.float64)) * scale
                context, _ = prescribed(matrix)
                sigma, u, v = min_singular_direction(context, iterations=20, solve_iterations=4, exact_max_elements=0)
                self.assertAlmostEqual(float(sigma / scale), 1., places=9)
                self.assertLess(float((matrix.T @ u - sigma * v).norm() / scale), 1e-4)
                self.assertEqual(context.numerical_diagnostics["min_svd"]["status"], "converged_estimate")

    def test_matrix_free_null_leakage_is_never_normalized(self):
        context, _ = prescribed(torch.diag(torch.tensor([1., 0.])))
        with self.assertRaisesRegex(RuntimeError, "null|converge"):
            min_singular_direction(context, exact_max_elements=0)

    def test_unconverged_matrix_free_output_raises(self):
        context, _ = prescribed(torch.diag(torch.tensor([2., 1.])))
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            min_singular_direction(context, iterations=1, exact_max_elements=0, tolerance=1e-8)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            max_singular_direction(context, iterations=1, exact_max_elements=0, tolerance=1e-8)

    def test_default_maximum_refines_within_cap_but_explicit_limit_is_hard(self):
        context, _ = prescribed(torch.diag(torch.tensor([1., .97, .8])))
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            max_singular_direction(context, iterations=10, exact_max_elements=0)
        torch.manual_seed(7)
        sigma, _, _ = max_singular_direction(context, exact_max_elements=0)
        self.assertAlmostEqual(float(sigma), 1., places=6)
        status = context.numerical_diagnostics["max_svd"]
        self.assertGreater(status["iterations"], 10)
        self.assertLessEqual(status["iterations"], 500)
        self.assertLessEqual(status["relative_residual"], 1e-4)

    def test_adaptive_maximum_handles_slower_spectral_gap_without_relaxing_tolerance(self):
        context, _ = prescribed(torch.diag(torch.tensor([1., .99, .8])))
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            max_singular_direction(context, exact_max_elements=0, max_iterations=200)
        torch.manual_seed(7)
        sigma, _, _ = max_singular_direction(context, exact_max_elements=0)
        self.assertAlmostEqual(float(sigma), 1., places=6)
        status = context.numerical_diagnostics["max_svd"]
        self.assertGreater(status["iterations"], 200)
        self.assertLessEqual(status["iterations"], 500)
        self.assertLessEqual(status["relative_residual"], 1e-4)

    def test_constructor_uses_same_configured_solver_as_ranking(self):
        context, package = prescribed(torch.diag(torch.tensor([.001, .000001])))
        settings = dict(iterations=20, solve_iterations=20, damping=1e-12)
        ranking = min_svd(package, context, **settings)[0]
        mask = [torch.ones_like(package.grads[0], dtype=torch.bool)]
        direction = min_svd_alignment(package, mask, context, **settings)[0]
        torch.testing.assert_close(ranking, direction.abs())
        self.assertGreater(float(abs(direction[1])), .999999)

    def test_constant_gradient_materializes_zero_derivatives(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
            def forward(self, x):
                return self.theta.sum().expand(x.shape[0])
        context = transformation_context(Constant(), [(torch.tensor([.2, .4], dtype=torch.float64), 0)], loss_function=lambda p, y: p.sum())
        vector = torch.ones(2, dtype=torch.float64)
        self.assertEqual(float(jacobian_vector_product(vector, context).norm()), 0.)
        self.assertEqual(float(jacobian_transpose_vector_product(vector, context).norm()), 0.)
        self.assertEqual(float(jtj_vector_product(vector, context).norm()), 0.)
        self.assertEqual(float(gradient_jacobian(context).norm()), 0.)

    def test_cg_handles_small_positive_operator(self):
        b = torch.tensor([1., 2.], dtype=torch.float64)
        actual = conjugate_gradient(lambda x: 1e-14 * x, b)
        torch.testing.assert_close(actual, b / 1e-14)

    def test_full_subspace_i2f_matches_regularized_inverse(self):
        matrix = torch.tensor([[1.7, .2], [.3, .6], [.2, -.1]], dtype=torch.float64)
        context, package = prescribed(matrix)
        actual = exact_i2f(package, context, samples=2, damping=.0001)[0]
        expected = (matrix @ torch.linalg.inv(matrix.T @ matrix + .0001 * torch.eye(2, dtype=matrix.dtype))).norm(dim=1)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            exact_i2f(package, context, damping=0.)

    def test_bound_solver_and_reference_norm_against_affine_case(self):
        context, package = prescribed(torch.diag(torch.tensor([2., 1.])))
        masks = [torch.ones(2, dtype=torch.bool)]
        noise = [torch.tensor([1., 0.], dtype=torch.float64)]
        settings = dict(target_error=.02, mu_j=0., mu_j_method="manual", max_scale=1., residual_fraction=0.)
        with contextlib.redirect_stdout(io.StringIO()):
            output = lower_bound_threshold(package, noise, masks, context, **settings)
            reference = max_svd_reference_scale(package, [noise[0].flip(0)], masks, context, **settings)
        expected_norm = math.sqrt(2 * .02) * 4 / 2
        self.assertAlmostEqual(float(output[0].norm()), expected_norm, places=12)
        self.assertAlmostEqual(float(reference[0].norm()), expected_norm, places=12)
        self.assertEqual(context.numerical_diagnostics["lower_bound"]["status"], "conditional_target_calibrated")
        self.assertFalse(context.numerical_diagnostics["lower_bound"]["certified"])
        json.dumps(context.numerical_diagnostics)
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(io.StringIO()):
            output = lower_bound_threshold(package, noise, masks, context, **{**settings, "max_scale": .01})
        self.assertAlmostEqual(float(output[0].norm() / package.grads[0].norm()), .01, places=12)
        self.assertEqual(len(caught), 1)
        self.assertEqual(context.numerical_diagnostics["lower_bound"]["status"], "infeasible_best_effort")


if __name__ == "__main__":
    unittest.main()
