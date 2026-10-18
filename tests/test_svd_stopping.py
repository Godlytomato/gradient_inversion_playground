"""Float32 power stopping must leave room for the final singular check."""
import unittest
from unittest.mock import patch

import torch

from transformations.transform_def import transformation_context
from transformations.ranking_implimentations.transform_utils import (
    max_singular_direction,
)


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

    def test_float32_proxy_can_pass_while_true_singular_residual_fails(self):
        image = self.matrix @ self.initial
        candidate = self.matrix.T @ image
        eigenvalue = self.initial @ candidate
        proxy = (candidate - eigenvalue * self.initial).norm() / eigenvalue
        sigma = image.norm()
        actual = (self.matrix.T @ (image / sigma) - sigma * self.initial).norm() / sigma
        self.assertLessEqual(float(proxy), self.tolerance)
        self.assertGreater(float(actual), self.tolerance)

        sigma, u, v = self.solve(100)
        residual = (self.matrix.T @ u - sigma * v).norm() / sigma
        self.assertLessEqual(float(residual), self.tolerance)
        diagnostic = self.context.numerical_diagnostics["max_svd"]
        self.assertGreater(diagnostic["iterations"], 1)
        self.assertLessEqual(diagnostic["iterations"], 100)

    def test_final_check_keeps_requested_tolerance_and_hard_budget(self):
        sigma, u, v = self.solve(1)
        residual = float((self.matrix.T @ u - sigma * v).norm() / sigma)
        # The internal margin may not require additional iterations beyond an
        # explicit cap, or tighten the user's accepted final residual.
        self.assertGreater(residual, self.tolerance * 0.5)
        self.assertLessEqual(residual, self.tolerance)
        self.assertEqual(self.context.numerical_diagnostics["max_svd"]["iterations"], 1)


if __name__ == "__main__":
    unittest.main()
