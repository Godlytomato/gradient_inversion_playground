import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.coalition_disguise.geometry import audit, coalitions, margins, observe, solve
from experiments.coalition_disguise.inputs import resolve_settings
from experiments.coalition_disguise.residual_solver import (
    ResidualSearchError, complement_lower_bound, project_shares, reference_basis, solve_residual)
from experiments.coalition_disguise.workflow import parse_args


class ResidualGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shorter_residual_can_have_more_nonzeros_and_must_be_rejected(self):
        g = torch.ones(4)
        d = torch.full((1, 4), .25)
        shares = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
        old, _ = audit(shares, g, d, .01)
        required, rows = audit(shares, g, d, .01, shares.double(), residual_nnz_gap=0)
        self.assertTrue(old['all_coalitions_pass'])
        self.assertFalse(required['all_coalitions_pass'])
        self.assertTrue(required['all_relative_distances_pass'])
        for row in rows:
            self.assertLess(row['public_distance'], row['private_distance'])
            self.assertEqual(row['disguise_residual_nnz'], 4)
            self.assertEqual(row['ground_truth_residual_nnz'], 2)
            self.assertFalse(row['residual_sparsity_pass'])

    def test_strict_simplicity_and_optional_equality_have_distinct_checks(self):
        g, d = torch.full((2,), 2.), torch.zeros(1, 2)
        shares = torch.ones(2, 2)
        equal, _ = audit(shares, g, d, 0, residual_nnz_gap=0)
        strict, _ = audit(shares, g, d, 0, residual_nnz_gap=1)
        self.assertTrue(equal['all_coalitions_pass'])
        self.assertFalse(strict['all_coalitions_pass'])
        d[:] = 1
        exact, rows = audit(shares, g, d, 0, residual_nnz_gap=2)
        self.assertTrue(exact['all_coalitions_pass'])
        self.assertTrue(all(r['disguise_residual_nnz'] == 0 and r['ground_truth_residual_nnz'] == 2 for r in rows))

    def test_sparsity_is_counted_after_float32_subset_reduction(self):
        shares = torch.tensor([[1e8], [-1e8+1], [-1e8], [1e8-1]], dtype=torch.float64)
        report, rows = audit(shares.float(), torch.zeros(1), torch.ones(1, 1), 0, shares, residual_nnz_gap=0)
        self.assertTrue(report['float32_aggregation_pass'])
        self.assertTrue(report['float64_aggregation_pass'])
        self.assertEqual(float(observe(shares, (0, 1))), 1)
        row = next(r for r in rows if r['coalition'] == 3)
        self.assertEqual(row['disguise_residual_nnz'], 1)
        self.assertEqual(row['ground_truth_residual_nnz'], 0)
        self.assertFalse(row['passed'])

    def test_projection_preserves_margins_aggregation_and_reduces_all_distances(self):
        rng = torch.Generator().manual_seed(91)
        g, other = torch.randn(2, 24, generator=rng, dtype=torch.float64)
        d = torch.stack([g, 2*g, other, other])
        shares = torch.randn(3, 24, generator=rng, dtype=torch.float64)
        shares[-1] = g-shares[:-1].sum(0)
        basis = reference_basis(g, d)
        self.assertEqual(basis.shape[1], 2)  # Redundant public references are safe.
        projected = project_shares(shares, g, basis)
        torch.testing.assert_close(projected.sum(0), g, rtol=1e-12, atol=1e-12)
        before = torch.stack([observe(shares, s) for s in coalitions(3)])
        after = torch.stack([observe(projected, s) for s in coalitions(3)])
        torch.testing.assert_close(margins(before, g, d), margins(after, g, d), rtol=1e-12, atol=1e-12)
        self.assertTrue(bool(((after[:, None]-d).square().sum(2)
                              <= (before[:, None]-d).square().sum(2)+1e-11).all()))

    def test_solver_returns_strictly_simpler_shorter_actual_observations(self):
        rng = torch.Generator().manual_seed(11)
        g = torch.randn(128, generator=rng)
        d = torch.randn(8, 128, generator=rng)
        result = solve_residual(g, d, seed=11, partitions=2, anchors=8, residual_nnz_gap=2)
        report, _ = audit(result['shares'], g, d, result['epsilon'], result['exact_shares'], residual_nnz_gap=2)
        self.assertTrue(report['all_coalitions_pass'])
        self.assertTrue(report['float32_aggregation_pass'])
        self.assertTrue(report['float64_aggregation_pass'])
        for subset in coalitions(3):
            h = observe(result['shares'], subset).double()
            distances = (d.double()-h).square().sum(1)
            rd, rg = h-d[int(distances.argmin())].double(), h-g.double()
            self.assertLessEqual(int(torch.count_nonzero(rd))+2, int(torch.count_nonzero(rg)))
            self.assertLessEqual(float(rd.square().sum())+result['epsilon'], float(rg.square().sum()))
        again = solve_residual(g, d, seed=11, partitions=2, anchors=8, residual_nnz_gap=2)
        self.assertTrue(torch.equal(result['shares'], again['shares']))

    def test_hard_absolute_cap_is_not_relaxed(self):
        g = torch.tensor([2., 2.])
        d = torch.ones(1, 2)
        good = solve_residual(g, d, n=2, partitions=1, max_disguise_distance_squared=0)
        self.assertTrue(good['audit']['all_coalitions_pass'])
        shares = torch.tensor([[.5, .5], [1.5, 1.5]])
        report, rows = audit(shares, g, d, 0, residual_nnz_gap=0, max_disguise_distance_squared=.49)
        self.assertFalse(report['all_absolute_distances_pass'])
        self.assertTrue(all(not r['absolute_distance_pass'] for r in rows))
        rng = torch.Generator().manual_seed(11)
        g = torch.randn(128, generator=rng)
        d = torch.randn(8, 128, generator=rng)
        with self.assertRaises(ResidualSearchError) as caught:
            solve_residual(g, d, seed=11, partitions=2, anchors=8, max_disguise_distance_squared=0)
        self.assertTrue(all(not any(r['absolute_distance_pass'] for r in a['coalition_geometry'])
                            for a in caught.exception.diagnostics['attempts']))

    def test_complement_distance_lower_bound(self):
        bound = complement_lower_bound(torch.tensor([2., 0.]), torch.zeros(1, 2))
        self.assertEqual(bound['complement_lower_bound_squared'], 1)
        exact = complement_lower_bound(torch.tensor([3., 3.]), torch.tensor([[1., 1.], [2., 2.]]))
        self.assertEqual(exact['complement_lower_bound_squared'], 0)
        self.assertEqual(set(exact['complement_public_slots']), {0, 1})

    def test_identical_public_and_true_gradients_cannot_meet_strict_simplicity(self):
        with self.assertRaisesRegex(ResidualSearchError, 'not an infeasibility proof') as caught:
            solve_residual(torch.ones(24), torch.ones(2, 24), partitions=1, residual_nnz_gap=1)
        self.assertTrue(caught.exception.diagnostics['attempts'])
        self.assertTrue(all(not a['all_coalitions_pass'] for a in caught.exception.diagnostics['attempts']))
        for gap in (-1, 1.5):
            with self.assertRaises(ValueError):
                solve_residual(torch.ones(2), torch.ones(1, 2), residual_nnz_gap=gap)

    def test_absolute_distance_scoring_takes_priority_over_share_sparsity(self):
        rng = torch.Generator().manual_seed(0)
        g, d = torch.randn(5, generator=rng), torch.randn(6, 5, generator=rng)
        sparse = solve(g, d, seed=0, partitions=2, anchors=5)
        distance = solve(g, d, seed=0, partitions=2, anchors=5, distance_first=True)
        self.assertLessEqual(max(r['public_distance'] for r in distance['geometry']),
                             max(r['public_distance'] for r in sparse['geometry']))

    def test_old_manifests_resolve_to_old_construction_and_reject_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = dict(model='lenet5', dataset='cifar10', private_index=193,
                       arguments=dict(seed=2026, public_count=100, servers=3,
                                      relative_margin=.01, partitions=8, anchors=16))
            (root/'manifest.json').write_text(json.dumps(old))
            args = parse_args(['--output', str(root), '--prepare-only'])
            resolve_settings(args)
            self.assertEqual(args.decomposition, 'sparse')
            self.assertEqual(args.private_index, 193)
            with self.assertRaisesRegex(ValueError, 'fresh --output'):
                resolve_settings(parse_args(['--output', str(root), '--decomposition', 'residual']))


if __name__ == '__main__':
    unittest.main()
