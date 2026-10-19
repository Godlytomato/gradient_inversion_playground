"""Rank-aware positive modes retain weak genuine directions and reject leakage."""
import unittest
from unittest.mock import patch

import torch

from run_utils.svd_settings import min_svd_parameters
from transformations.positive_svd import positive_singular_direction
from transformations.ranking_implimentations.transform_utils import batch_tensors
from transformations.transform_def import transformation_context


class GradientMap(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.ones(matrix.shape[0], dtype=matrix.dtype))
        self.register_buffer("matrix", matrix)

    def forward(self, inputs):
        return ((inputs.flatten(1) @ self.matrix.T) * self.theta).sum(1)


class SumLoss(torch.nn.Module):
    def forward(self, prediction, labels):
        return prediction.sum()


def geometry(matrix):
    return transformation_context(
        GradientMap(matrix).eval(), [(torch.ones(matrix.shape[1], dtype=matrix.dtype), 0)],
        loss_function=SumLoss().eval(),
    )


class PositiveSVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def solve(self, matrix, tolerance=1e-4, **parameters):
        context = geometry(matrix)
        images, labels = batch_tensors(context)
        settings = min_svd_parameters(**parameters)
        triplet = positive_singular_direction(context, images, labels, settings, tolerance)
        return context, triplet

    def test_rotated_rectangular_rank_deficient_map_matches_dense_svd(self):
        generator = torch.Generator().manual_seed(71)
        left, _ = torch.linalg.qr(torch.randn(7, 4, generator=generator, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
        matrix = left @ torch.diag(torch.tensor([2., .7, .03, 0.], dtype=torch.float64)) @ right.T
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                context, (sigma, u, v) = self.solve(matrix.to(dtype))
                self.assertAlmostEqual(float(sigma), .03, places=6)
                current = matrix.to(dtype)
                self.assertLess(float((current @ v - sigma * u).norm() / 2), 1e-6)
                self.assertLess(float((current.T @ u - sigma * v).norm() / 2), 1e-5)
                diagnostic = context.numerical_diagnostics["min_svd"]
                self.assertEqual(diagnostic["numerical_rank"], 3)
                self.assertEqual(diagnostic["input_nullity"], 1)
                self.assertEqual(diagnostic["selected_index"], 2)
                self.assertEqual(diagnostic["method"], "dense_qr_svd")

    def test_wide_map_includes_implicit_input_nullspace(self):
        matrix = torch.tensor([[2., 0., 0., 0.], [0., .1, 0., 0.]], dtype=torch.float64)
        context, (sigma, _, _) = self.solve(matrix)
        self.assertAlmostEqual(float(sigma), .1)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["input_nullity"], 2)

    def test_zero_operator_has_no_positive_mode(self):
        with self.assertRaisesRegex(RuntimeError, "No positive"):
            self.solve(torch.zeros(4, 3, dtype=torch.float64))

    def test_cutoff_equality_is_excluded_and_cutoff_is_independent_of_residual(self):
        matrix = torch.diag(torch.tensor([1., .1, .01, 0.], dtype=torch.float64))
        for tolerance in (1e-3, 1e-8):
            context, (sigma, _, _) = self.solve(
                matrix, tolerance=tolerance, relative_rank_tolerance=.01,
            )
            self.assertAlmostEqual(float(sigma), .1)
            self.assertEqual(context.numerical_diagnostics["min_svd"]["numerical_rank"], 2)

    def test_selection_and_rank_are_invariant_under_scalar_rescaling(self):
        for scale in (1e-8, 1., 1e8):
            with self.subTest(scale=scale):
                matrix = torch.diag(torch.tensor([1., .02, 0.], dtype=torch.float64)) * scale
                context, (sigma, _, _) = self.solve(matrix)
                self.assertAlmostEqual(float(sigma / scale), .02, places=10)
                self.assertEqual(context.numerical_diagnostics["min_svd"]["numerical_rank"], 2)

    def test_explicit_cutoff_accepts_weak_float32_mode_below_sqrt_epsilon(self):
        matrix = torch.diag(torch.tensor([1., 2e-5, 0.], dtype=torch.float32))
        context, (sigma, u, v) = self.solve(matrix)
        self.assertAlmostEqual(float(sigma), 2e-5, places=10)
        self.assertGreater(float(u[1].abs()), .99999)
        self.assertGreater(float(v[1].abs()), .99999)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["numerical_rank"], 2)

    def test_dense_left_vector_is_not_replaced_by_live_jvp(self):
        matrix = torch.diag(torch.tensor([1., 2e-5, 0.], dtype=torch.float32))
        from transformations.ranking_implimentations import transform_utils
        original = transform_utils.jacobian_vector_product

        def rounded_jvp(vector, context):
            result = original(vector, context)
            result[0] += 1e-8
            return result

        # A tiny, acceptable JVP error would become a visible null/large-mode
        # component if the weak left vector were reconstructed by normalization.
        with patch.object(transform_utils, "jacobian_vector_product", side_effect=rounded_jvp):
            _, (_, u, _) = self.solve(matrix)
        self.assertLess(float(u[0].abs()), 1e-8)
        self.assertGreater(float(u[1].abs()), .99999)

    def test_independent_operator_validation_rejects_wrong_dense_backend(self):
        matrix = torch.diag(torch.tensor([2., .5], dtype=torch.float64))

        def wrong_backend(context, images, labels, probe, diagnostic):
            return lambda vector: 2 * matrix @ vector, lambda vector: 2 * matrix.T @ vector

        with patch("transformations.minimum_krylov._derivative_products", side_effect=wrong_backend):
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                self.solve(matrix)

    def test_declared_dense_cap_switches_to_bounded_matrix_free_solver(self):
        matrix = torch.diag(torch.tensor([2., .5], dtype=torch.float64))

        def bounded_failure(*args, **kwargs):
            self.assertEqual(kwargs["max_iterations"], 1)
            self.assertEqual(kwargs["rank_policy"], "smallest_nonzero")
            raise RuntimeError("declared Krylov cap exhausted")

        with patch("transformations.minimum_krylov.minimum_krylov", side_effect=bounded_failure):
            with self.assertRaisesRegex(RuntimeError, "cap exhausted"):
                self.solve(matrix, dense_max_elements=1, krylov_max_iterations=1)


if __name__ == "__main__":
    unittest.main()
