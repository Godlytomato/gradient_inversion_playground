import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.coalition_disguise.geometry import coalitions, margins, solve, audit, observe
from experiments.coalition_disguise.run import claim_manifest, build_attack, attack, gradient
from models.lenet import LeNet5


class CoalitionDisguiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_proper_coalitions(self):
        for n in range(2, 6):
            subsets = coalitions(n)
            self.assertEqual(len(subsets), 2**n - 2)
            self.assertEqual(len(set(subsets)), len(subsets))
            self.assertTrue(all(0 < len(s) < n for s in subsets))

    def test_distance_cancellation_identity(self):
        rng = torch.Generator().manual_seed(7)
        g, d, h = [torch.randn(*shape, generator=rng).double() for shape in [(9,), (5, 9), (6, 9)]]
        direct = (h - g).square().sum(1)[:, None] - (h[:, None] - d).square().sum(2)
        torch.testing.assert_close(margins(h, g, d), direct)

    def test_feasible_partition_no_repair(self):
        d = torch.tensor([[float(mask & (1 << i) != 0) for i in range(5)] for mask in range(31)])
        result = solve(torch.ones(5), d, relative_margin=.01)
        self.assertEqual(result['audit']['extra_nnz'], 0)
        self.assertIsNone(result['repair_coordinate'])
        self.assertEqual(result['audit']['float64_max_error'], 0)
        self.assertTrue(result['audit']['all_coalitions_pass'])

    def test_one_extra_nonzero_repair(self):
        rng = torch.Generator().manual_seed(0)
        g, d = torch.randn(5, generator=rng), torch.randn(6, 5, generator=rng)
        result = solve(g, d, seed=0, partitions=2, anchors=5)
        self.assertEqual(result['audit']['extra_nnz'], 1)
        self.assertEqual(result['alpha'].sum(), 0)
        self.assertEqual(result['audit']['float64_max_error'], 0)
        self.assertTrue(result['audit']['all_coalitions_pass'])
        torch.testing.assert_close(result['shares'].sum(0), g)

    def test_float32_rounding_is_audited(self):
        exact = torch.tensor([[1e8], [-1e8 + 1], [0]], dtype=torch.float64)
        g, d = torch.ones(1), torch.zeros(1, 1)
        report, _ = audit(exact.float(), g, d, 0, exact)
        self.assertEqual(report['float64_max_error'], 0)
        self.assertEqual(report['float32_max_error'], 1)
        self.assertFalse(report['float32_aggregation_pass'])

    def test_raw_sums_no_rescaling(self):
        shares = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        self.assertTrue(torch.equal(observe(shares, (0, 2)), torch.tensor([6., 8.])))

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            claim_manifest(directory, dict(seed=1))
            original = (directory / 'manifest.json').read_bytes()
            claim_manifest(directory, dict(seed=1))
            self.assertEqual(original, (directory / 'manifest.json').read_bytes())
            with self.assertRaises(ValueError):
                claim_manifest(directory, dict(seed=2))
            self.assertEqual(original, (directory / 'manifest.json').read_bytes())

    def args(self, cap=None):
        return argparse.Namespace(seed=2026, restarts=1, lr=None, patience=2,
                                  minimum_change=1e-7, max_iterations=cap)

    def test_stall_only_adapter_preserves_shared_validation(self):
        for name in ('idlg', 'dlg', 'geiping'):
            executable = build_attack(name, self.args())
            self.assertEqual([p.name for p in executable.observer.fail_policies], ['no_improvement'])
            self.assertFalse(executable.requirements.attack_information & {'ground_truth', 'original_gradients'})
            self.assertEqual(executable.observer.best_selection.metric, 'attack_loss')
        self.assertIn('max_iterations', [p.name for p in build_attack('idlg', self.args(2)).observer.fail_policies])

    def test_real_preset_attack_and_completed_resume(self):
        torch.manual_seed(11)
        model = LeNet5()
        model.conv1 = torch.nn.Conv2d(3, 6, 5, padding=2)
        image, label = torch.rand(1, 3, 32, 32), torch.tensor([1])
        model(image)
        model.eval()
        g = gradient(model, image, label)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            summary, result = attack(model, g, label, 'geiping', self.args(2), path)
            self.assertEqual(summary['reason'], 'max_iterations')
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(summary['total_iterations'], 2)
            self.assertTrue(torch.isfinite(result).all())
            # Main commits completion only after evaluation; simulate that commit.
            (path / 'summary.json').write_text(json.dumps(summary))
            with patch('experiments.coalition_disguise.run.build_attack', side_effect=AssertionError('reran completed job')):
                saved, resumed = attack(model, g, label, 'geiping', self.args(2), path)
            self.assertEqual(saved, summary)
            self.assertTrue(torch.equal(result, resumed))
            torch.save(torch.zeros_like(result), path / 'reconstruction.pt')
            with self.assertRaises(ValueError):
                attack(model, g, label, 'geiping', self.args(2), path)

    def test_exhausted_search_is_not_infeasibility_claim(self):
        with self.assertRaisesRegex(RuntimeError, 'not an infeasibility proof'):
            solve(torch.zeros(2), torch.ones(1, 2), partitions=1)


if __name__ == '__main__':
    unittest.main()
