"""Minimum-mode estimates against dense SVDs and unresolved subspaces."""
import unittest
from unittest.mock import patch

import torch

from transformations.transform_def import grad_package, transformation_context
from transformations.ranking_implimentations.transform_utils import min_singular_direction
from transformations.ranking_implimentations.min_svd import min_svd
from transformations.transformation_implimentations.min_svd_alignment import min_svd_alignment


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
    model = GradientMap(matrix).eval()
    image = torch.ones(matrix.shape[1], dtype=matrix.dtype)
    return transformation_context(
        model, [(image, 0)], loss_function=SumLoss().eval(),
    ), grad_package([matrix @ image])


class MinimumKrylovTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def solve(self, context, **settings):
        return min_singular_direction(
            context, solver_method="lanczos_v1", exact_max_elements=0, **settings,
        )

    def test_dense_rotated_maps_match_smallest_singular_triplet(self):
        generator = torch.Generator().manual_seed(27)
        left, _ = torch.linalg.qr(torch.randn(7, 4, generator=generator, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
        matrix = left @ torch.diag(torch.tensor([2., .7, .2, .03], dtype=torch.float64)) @ right.T
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                current = matrix.to(dtype)
                context, _ = geometry(current)
                sigma, u, v = self.solve(context)
                expected = torch.linalg.svdvals(current.double())[-1]
                self.assertAlmostEqual(float(sigma), float(expected), places=6)
                self.assertLess(float((current.T @ u - sigma * v).norm() / 2), 1e-4)
                self.assertLess(float((current @ v - sigma * u).norm() / 2), 1e-6)
                self.assertEqual(context.numerical_diagnostics["min_svd"]["iterations"], 4)

    def test_clustered_bottom_spectrum_and_scalar_rescaling(self):
        for scale in (1e-8, 1., 1e8):
            with self.subTest(scale=scale):
                matrix = torch.diag(torch.tensor([2., .4, .01001, .01], dtype=torch.float64)) * scale
                context, _ = geometry(matrix)
                sigma, _, v = self.solve(context)
                self.assertAlmostEqual(float(sigma / scale), .01, places=10)
                self.assertGreater(float(v[-1].abs()), .999999)

    def test_nullspace_is_rejected_and_never_normalized_into_noise(self):
        for diagonal in ([1., .2, 0.], [0., 0., 0.]):
            context, _ = geometry(torch.diag(torch.tensor(diagonal, dtype=torch.float64)))
            with self.subTest(diagonal=diagonal), self.assertRaisesRegex(RuntimeError, "null|nonzero"):
                self.solve(context)

    def test_explicit_work_and_memory_caps_are_hard(self):
        context, _ = geometry(torch.diag(torch.tensor([2., 1.], dtype=torch.float64)))
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            self.solve(context, krylov_max_iterations=1, tolerance=1e-8)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["iterations"], 1)
        with self.assertRaisesRegex(RuntimeError, "elements"):
            self.solve(context, krylov_max_elements=1)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["status"], "resource_cap")

    def test_invariant_start_explores_missing_minimum_before_accepting(self):
        matrix = torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64))
        context, _ = geometry(matrix)
        e1, e2, e3 = torch.eye(3, dtype=torch.float64)
        with patch("torch.randn", side_effect=[e1, e1, e2, e3]):
            sigma, _, v = self.solve(context)
        self.assertAlmostEqual(float(sigma), 1., places=12)
        self.assertGreater(float(abs(v[-1])), .999999)
        context, _ = geometry(matrix)
        with patch("torch.randn", side_effect=[e1, e1]):
            with self.assertRaisesRegex(RuntimeError, "invariant subspace"):
                self.solve(context, krylov_max_iterations=1)

    def test_solver_stream_is_independent_and_cached_mode_is_reused(self):
        matrix = torch.diag(torch.tensor([1., .99999, .99998], dtype=torch.float64))
        context, package = geometry(matrix)
        settings = {"solver_method": "lanczos_v1", "exact_max_elements": 0}
        ranking, = min_svd(package, context, **settings)
        torch.randn(100)
        masks = [torch.ones(3, dtype=torch.bool)]
        # Construction must reuse the checked ranking mode, even if the solver
        # could otherwise return another vector from a clustered eigenspace.
        with patch("transformations.minimum_krylov.minimum_krylov", side_effect=AssertionError("duplicate solve")):
            noise, = min_svd_alignment(package, masks, context, **settings)
        torch.testing.assert_close(noise.abs(), ranking)
        fresh, _ = geometry(matrix)
        _, fresh_u, _ = self.solve(fresh)
        torch.testing.assert_close(fresh_u, noise)

    def test_forward_ad_fallback_keeps_independent_final_validation(self):
        context, _ = geometry(torch.diag(torch.tensor([2., 1.], dtype=torch.float64)))
        with patch("torch.func.jvp", side_effect=NotImplementedError("custom op lacks forward AD")):
            sigma, _, _ = self.solve(context)
        self.assertAlmostEqual(float(sigma), 1., places=12)
        self.assertEqual(context.numerical_diagnostics["min_svd"]["derivative_backend"], "autograd")


if __name__ == "__main__":
    unittest.main()
