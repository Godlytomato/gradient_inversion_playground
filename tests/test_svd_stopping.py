"""Check the actual singular residual using accurate large-vector norms."""
import unittest
from unittest.mock import patch

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.ranking_implimentations.transform_utils import (
    jacobian_vector_product, jacobian_transpose_vector_product,
    max_singular_direction, min_singular_direction,
)
from transformations.transformation_implimentations.max_svd_alignment import max_svd_alignment
from transformations.transformation_implimentations.min_svd_alignment import min_svd_alignment
from transformations.transformation_metrics import measure_transformation


class LinearGradientMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(matrix.shape[0]))
        self.register_buffer("matrix", matrix)

    def forward(self, images):
        return ((images.flatten(1) @ self.matrix.T) * self.theta).sum(1)


class MaximumStoppingTests(unittest.TestCase):
    def setUp(self):
        # This rotated, well-conditioned Jacobian reproduces real float32
        # disagreement at the stopping boundary, without perturbing operators.
        self.matrix = torch.tensor([
            [0.9553365111351013, 0.29552021622657776],
            [-0.28665462136268616, 0.9266764521598816],
        ], dtype=torch.float32)
        self.initial = torch.tensor([
            0.9548352956771851, 0.2971356511116028,
        ], dtype=torch.float32)
        self.context = transformation_context(
            LinearGradientMap(self.matrix), [(torch.ones(2), 0)],
            loss_function=lambda predictions, labels: predictions.sum(),
        )
        self.tolerance = 1e-4

    def solve(self, budget):
        with patch(
            "transformations.ranking_implimentations.transform_utils._random_unit",
            return_value=self.initial,
        ):
            return max_singular_direction(
                self.context, iterations=1, max_iterations=budget,
                tolerance=self.tolerance, exact_max_elements=0,
            )

    def test_boundary_uses_accurate_norm_and_requested_residual(self):
        image = self.matrix @ self.initial
        candidate = self.matrix.T @ image
        eigenvalue = self.initial @ candidate
        proxy = (candidate - eigenvalue * self.initial).norm() / eigenvalue
        sigma = image.norm()
        actual = (self.matrix.T @ (image / sigma) - sigma * self.initial).norm() / sigma
        self.assertLessEqual(float(proxy), self.tolerance)
        self.assertGreater(float(actual), self.tolerance)

        sigma, u, v = self.solve(100)
        residual = torch.linalg.vector_norm(
            self.matrix.T @ u - sigma * v, dtype=torch.float64,
        ) / sigma
        self.assertLessEqual(float(residual), self.tolerance)
        diagnostic = self.context.numerical_diagnostics["max_svd"]
        # Accurate normalization may legitimately satisfy the tolerance at
        # the initial iterate; a biased float32 norm used to require another.
        self.assertGreaterEqual(diagnostic["iterations"], 1)
        self.assertLessEqual(diagnostic["iterations"], 100)

    def test_final_check_keeps_requested_tolerance_and_hard_budget(self):
        sigma, u, v = self.solve(1)
        residual = float((self.matrix.T @ u - sigma * v).norm() / sigma)
        # Preserve the accepted final tolerance and explicit iteration cap.
        self.assertGreater(residual, self.tolerance * 0.5)
        self.assertLessEqual(residual, self.tolerance)
        self.assertEqual(self.context.numerical_diagnostics["max_svd"]["iterations"], 1)


class RankOneGradientMap(torch.nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones_like(coefficients))
        self.register_buffer("coefficients", coefficients)

    def forward(self, images):
        return (images.flatten(1)[:, :1] * self.coefficients * self.theta).sum(1)


class LargeGradientStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        # A compact witness for the norm error in an 11M-parameter ResNet.
        # The exact Jacobian is rank one: first column=coefficients, second=0.
        self.coefficients = torch.full((65536,), .003)
        self.coefficients[:4] = 1.
        self.initial = torch.tensor([1., 0.])
        self.context = self.make_context(2)
        self.expected_sigma = torch.linalg.vector_norm(self.coefficients, dtype=torch.float64)

    def make_context(self, input_size):
        return transformation_context(
            RankOneGradientMap(self.coefficients), [(torch.ones(input_size), 0)],
            loss_function=lambda predictions, labels: predictions.sum(),
        )

    def solve_max(self, tolerance=1e-4, cap=10):
        with patch(
            "transformations.ranking_implimentations.transform_utils._random_unit",
            return_value=self.initial,
        ):
            return max_singular_direction(
                self.context, iterations=1, max_iterations=cap,
                tolerance=tolerance, exact_max_elements=0,
            )

    def test_large_gradient_norm_does_not_create_false_nonconvergence(self):
        image = jacobian_vector_product(self.initial, self.context)
        biased_sigma = image.norm()
        biased_residual = (
            jacobian_transpose_vector_product(image / biased_sigma, self.context)
            - biased_sigma * self.initial
        ).norm() / biased_sigma
        self.assertGreater(float(biased_residual), 1e-4)

        sigma, u, v = self.solve_max()
        torch.testing.assert_close(sigma, self.expected_sigma)
        relative = torch.linalg.vector_norm(
            jacobian_transpose_vector_product(u, self.context) - sigma * v,
            dtype=torch.float64,
        ) / sigma
        self.assertLess(float(relative), 1e-4)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(u, dtype=torch.float64)), 1., places=6)

    def test_actual_residual_consumes_budget_even_when_normal_equations_proxy_is_zero(self):
        image = jacobian_vector_product(self.initial, self.context)
        candidate = jacobian_transpose_vector_product(image, self.context)
        eigenvalue = self.initial @ candidate
        self.assertEqual(float((candidate - eigenvalue * self.initial).norm()), 0.)
        # This accuracy is below the float32 operator's residual floor. A zero
        # proxy cannot terminate the solve early or override the final check.
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.solve_max(tolerance=1e-8, cap=3)
        diagnostic = self.context.numerical_diagnostics["max_svd"]
        self.assertEqual(diagnostic["iterations"], 3)
        self.assertGreater(diagnostic["relative_residual"], 1e-8)

    def test_minimum_uses_accurate_gradient_norm_too(self):
        # One input coordinate makes this a full-rank minimum problem.
        context = self.make_context(1)
        sigma, u, v = min_singular_direction(context, exact_max_elements=0)
        torch.testing.assert_close(sigma, self.expected_sigma)
        relative = torch.linalg.vector_norm(
            jacobian_transpose_vector_product(u, context) - sigma * v,
            dtype=torch.float64,
        ) / sigma
        self.assertLess(float(relative), 1e-4)

    def test_alignment_normalization_preserves_unit_norm_after_masking(self):
        unit = self.coefficients / self.expected_sigma
        package = grad_package([self.coefficients])
        for alignment, name in ((max_svd_alignment, "max"), (min_svd_alignment, "min")):
            for selected_size in (len(unit), 32768):
                with self.subTest(alignment=name, selected_size=selected_size):
                    mask = torch.arange(len(unit)) < selected_size
                    with patch(
                        "transformations.transformation_implimentations."
                        + name + "_svd_alignment." + name + "_singular_direction",
                        return_value=(self.expected_sigma, unit, self.initial),
                    ):
                        noise, = alignment(package, [mask], self.context)
                    norm = torch.linalg.vector_norm(noise, dtype=torch.float64)
                    self.assertAlmostEqual(float(norm), 1., places=6)
                    self.assertEqual(float(noise[~mask].abs().sum()), 0.)

    def test_reported_perturbation_norm_uses_accurate_accumulation(self):
        metrics = measure_transformation(
            [self.coefficients], grad_package([2 * self.coefficients]),
        )
        self.assertAlmostEqual(metrics["delta_l2"], float(self.expected_sigma), places=10)
        self.assertAlmostEqual(metrics["relative_delta_l2"], 1., places=10)


if __name__ == "__main__":
    unittest.main()
