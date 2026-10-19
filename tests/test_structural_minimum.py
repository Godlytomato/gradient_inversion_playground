"""A proven input nullspace must fail before any singular iteration."""
import unittest
from unittest.mock import patch

import torch

from models.lenet import LeNet5, change_channels
from transformations.transform_def import transformation_context
from transformations.ranking_implimentations.transform_utils import min_singular_direction


class StructuralMinimumTests(unittest.TestCase):
    def context(self):
        model = LeNet5().eval()
        change_channels(model, 3)
        return transformation_context(model, [(torch.zeros(3, 32, 32), 0)])

    def test_both_solvers_reject_proven_zero_minimum_without_iteration(self):
        for method in ("inverse_v2", "lanczos_v1"):
            with self.subTest(method=method):
                context = self.context()
                rng = torch.random.get_rng_state().clone()
                with patch(
                    "transformations.ranking_implimentations.transform_utils._small_jacobian",
                    side_effect=AssertionError("unnecessary derivative evaluation"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "rank <= 1992 < 3072"):
                        min_singular_direction(context, solver_method=method)
                diagnostic = context.numerical_diagnostics["min_svd"]
                self.assertEqual(diagnostic["status"], "structural_input_nullspace")
                self.assertEqual(diagnostic["solver_method"], method)
                self.assertTrue(diagnostic["rank_deficiency_proven"])
                self.assertEqual(diagnostic["structural_rank"]["nullity_lower_bound"], 1080)
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_explicit_minimum_positive_policy_keeps_its_separate_resource_checks(self):
        context = self.context()
        with self.assertRaisesRegex(RuntimeError, "smallest_nonzero rank policy requires exact SVD"):
            min_singular_direction(context, rank_policy="smallest_nonzero", solver_method="inverse_v2")
        self.assertNotIn("min_svd", context.numerical_diagnostics)


if __name__ == "__main__":
    unittest.main()
