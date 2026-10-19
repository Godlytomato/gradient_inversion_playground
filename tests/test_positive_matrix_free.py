"""A minimum-positive search must resolve rank rather than skip weak modes."""
import unittest
from unittest.mock import patch

import torch

from test_positive_svd import SumLoss, geometry
from transformations.transform_def import transformation_context
from transformations.ranking_implimentations.transform_utils import min_singular_direction


class PositiveMatrixFreeTests(unittest.TestCase):
    def solve(self, matrix, **kwargs):
        context = geometry(matrix)
        result = min_singular_direction(context, dense_max_elements=0, **kwargs)
        return context, result

    def test_complete_projection_matches_dense_minimum_positive_with_multiple_nulls(self):
        generator = torch.Generator().manual_seed(87)
        rotation, _ = torch.linalg.qr(torch.randn(5, 5, generator=generator, dtype=torch.float64))
        matrix = torch.diag(torch.tensor([2., .5, .03, 0., 0.], dtype=torch.float64)) @ rotation.T
        context, (sigma, u, v) = self.solve(matrix)
        self.assertAlmostEqual(float(sigma), .03, places=10)
        self.assertLess(float((matrix.T @ u - sigma * v).norm()), 1e-10)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["numerical_rank"], 3)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["input_nullity"], 2)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["rank_resolution"], "complete_input_projection")

    def test_cutoff_excludes_weak_nonzero_modes_consistently_under_rescaling(self):
        for scale in (1e-8, 1., 1e8):
            with self.subTest(scale=scale):
                matrix = torch.diag(torch.tensor([1., .04, .0005, 0.], dtype=torch.float64)) * scale
                context, (sigma, _, _) = self.solve(matrix, relative_rank_tolerance=.001)
                self.assertAlmostEqual(float(sigma / scale), .04, places=10)
                self.assertEqual(context.numerical_diagnostics["min_svd"]["numerical_rank"], 2)

    def test_partial_null_subspace_cannot_substitute_an_interior_positive_mode(self):
        matrix = torch.diag(torch.tensor([3., 2., 1., 0.], dtype=torch.float64))
        context = geometry(matrix)
        e1, _, _, e4 = torch.eye(4, dtype=torch.float64)
        # The two-vector subspace spans a null direction and the maximum mode,
        # missing both smaller positive modes. Filtering its spectrum and
        # accepting the maximum would be a scientifically different direction.
        with patch("torch.randn", side_effect=[e1, (e1 + e4) / 2**.5]):
            with self.assertRaisesRegex(RuntimeError, "invariant subspace|partial input subspace"):
                min_singular_direction(context, dense_max_elements=0, krylov_max_iterations=2)
        self.assertNotEqual(context.numerical_diagnostics["min_svd"]["status"], "converged_minimum_positive_estimate")

    def test_spurious_positive_ritz_value_is_rejected_by_original_operator(self):
        actual = torch.diag(torch.tensor([1., .02, 0.], dtype=torch.float64))
        perturbed = torch.diag(torch.tensor([1., .02, .001], dtype=torch.float64))

        def wrong_backend(context, images, labels, probe, diagnostic):
            return lambda vector: perturbed @ vector, lambda vector: perturbed.T @ vector

        with patch("transformations.minimum_krylov._derivative_products", side_effect=wrong_backend):
            with self.assertRaisesRegex(RuntimeError, "rank cutoff"):
                self.solve(actual)

    def test_legacy_zero_dense_cap_still_disables_dense_positive_route(self):
        matrix = torch.diag(torch.tensor([2., .2, 0.], dtype=torch.float64))
        context = geometry(matrix)
        with patch("transformations.positive_svd._dense", side_effect=AssertionError("dense is disabled")):
            sigma, _, _ = min_singular_direction(context, exact_max_elements=0)
        self.assertAlmostEqual(float(sigma), .2, places=10)

    def test_default_direct_api_initializes_lazy_parameters_before_dense_count(self):
        model = torch.nn.LazyLinear(2, dtype=torch.float64).eval()
        context = transformation_context(
            model, [(torch.ones(3, dtype=torch.float64), 0)], loss_function=SumLoss(),
        )
        sigma, _, _ = min_singular_direction(context)
        self.assertAlmostEqual(float(sigma), 2**.5, places=10)
        self.assertTrue(context.numerical_diagnostics["min_svd"]["lazy_initialized_for_operator"])


if __name__ == "__main__":
    unittest.main()
