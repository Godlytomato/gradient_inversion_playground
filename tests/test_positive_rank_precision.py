"""Normal-equation roundoff must not silently remove a genuine positive mode."""
import unittest

import torch

from transformations.ranking_implimentations.transform_utils import min_singular_direction
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


class PositiveRankPrecisionTests(unittest.TestCase):
    def test_float32_full_basis_cannot_skip_an_unresolved_positive_mode(self):
        generator = torch.Generator().manual_seed(22)
        left, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
        right, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator, dtype=torch.float64))
        matrix = (left @ torch.diag(torch.tensor([1., .001, .00002, 0.], dtype=torch.float64)) @ right.T).float()
        context = transformation_context(
            GradientMap(matrix).eval(), [(torch.ones(4), 0)], loss_function=SumLoss().eval(),
        )
        values = torch.linalg.svdvals(matrix.double())
        self.assertGreater(float(values[2]), 1e-5 * float(values[0]))
        self.assertLess(float(values[3]), 1e-5 * float(values[0]))
        # The old search gave the .001 mode a passing residual after a negative
        # Ritz value hid the genuine .00002 mode, and incorrectly reported rank 2.
        with self.assertRaisesRegex(RuntimeError, "precision|rank.*resolv|resolv.*rank"):
            min_singular_direction(
                context, exact_max_elements=0,
                krylov_max_iterations=4, krylov_check_every=1,
            )
        self.assertEqual(
            context.numerical_diagnostics["min_svd"]["status"],
            "rank_unresolved_at_operator_precision",
        )


if __name__ == "__main__":
    unittest.main()
