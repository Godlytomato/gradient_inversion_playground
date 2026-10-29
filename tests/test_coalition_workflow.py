import argparse
from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from data.dataset_def import dataset_def
from data.dataset_registry import dataset_dict
from models.model_def import model_definition
from models.model_registry import model_dict
from experiments.coalition_disguise.common import digest, gradient, model_digest
from experiments.coalition_disguise.geometry import audit
from experiments.coalition_disguise.inputs import build_model, select_indices
from experiments.coalition_disguise.workflow import main, evaluate_result, subset_plan
from experiments.coalition_disguise.residual_solver import ResidualSearchError


class SeededCoalitionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        rng = torch.Generator().manual_seed(47)
        self.private = TensorDataset(.1+.8*torch.rand(10, 3, 2, 2, generator=rng), torch.arange(10))
        self.public = TensorDataset(.1+.8*torch.rand(30, 3, 2, 2, generator=rng), torch.arange(30) % 10)
        self.data = dataset_def('toy', lambda: self.private, 10, 3, (2, 2), input_space='raw_pixels', split='test')
        self.model = model_definition('tiny', 3, (2, 2), 10,
            lambda: torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10)), None, None, None)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(dataset_dict, {'toy': self.data}))
        self.stack.enter_context(patch.dict(model_dict, {'tiny': self.model}))
        self.stack.enter_context(patch('experiments.coalition_disguise.inputs.load_datasets', return_value=(self.private, self.public)))

    def argv(self, root, *extra):
        return ['--model', 'tiny', '--dataset', 'toy', '--seed', '2026', '--output', str(root),
                '--public-count', '8', '--partitions', '2', '--anchors', '8', '--restarts', '1',
                '--patience', '3', '--max-iterations', '20', *extra]

    def test_sample_and_public_indices_are_seeded_and_independent(self):
        a = select_indices(10000, 50000, 100, 2026)
        torch.rand(1000)
        b = select_indices(10000, 50000, 100, 2026)
        c = select_indices(10000, 50000, 100, 2027)
        self.assertEqual(a, b)
        self.assertNotEqual(a[0], c[0])
        self.assertNotEqual(a[1], c[1])
        self.assertEqual(len(set(a[1])), 100)
        self.assertEqual(select_indices(10000, 50000, 100, 2026, private_index=193)[0], 193)

    def test_fixed_disguise_is_chosen_by_gradient_before_image_evaluation(self):
        model = build_model('tiny', self.data, self.private[0][0][None], 2026)
        x = torch.zeros(1, 3, 2, 2)
        public = torch.stack([torch.ones(3, 2, 2), torch.full((3, 2, 2), .2)])
        y = torch.tensor([2])
        g = gradient(model, x, y)
        d = torch.stack([gradient(model, p[None], y) for p in public])
        shares = torch.stack([d[0], g - d[0]])
        artifact = dict(g=g, d=d, shares=shares)
        _, rows = audit(shares, g, d, 0)
        plans = subset_plan(artifact, dict(coalition_geometry=rows), [7, 8], [2, 2])
        assigned = plans[0]
        self.assertEqual(assigned['disguise_index'], 7)
        # The reconstruction is exactly the OTHER public image. It must not
        # change the disguise selected from the observation's gradient distance.
        z = public[1][None]
        result = evaluate_result(model, z, x, public, y, shares[0], artifact, [7, 8],
                                 dict(selected_label=2), assigned)
        self.assertEqual(result['disguise_index'], 7)
        self.assertAlmostEqual(result['disguise_mse'], .64, places=6)
        self.assertAlmostEqual(result['ground_truth_mse'], .04, places=6)
        self.assertNotIn('public_min_mse', result)

    def test_full_workflow_runs_one_baseline_then_all_six_fixed_subsets_and_resumes(self):
        from experiments.coalition_disguise.attacks import attack as real_attack
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            calls = []
            def capture(*args, **kwargs):
                destination = args[5]
                calls.append(destination.name)
                plan = json.loads((root/'subsets_v3.json').read_text())
                self.assertEqual(len(plan), 6)  # Assignment exists before first attack.
                return real_attack(*args, **kwargs)
            with patch('experiments.coalition_disguise.workflow.attack', side_effect=capture):
                status = main(self.argv(root))
            self.assertEqual(calls, ['full', '001', '010', '011', '100', '101', '110'])
            self.assertEqual(status[0]['state'], 'complete')
            self.assertEqual(status[0]['completed_subsets'], 6)
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(len(report), 7)
            self.assertEqual(report[0]['coalition'], 'full')
            self.assertIsNone(report[0]['disguise_index'])
            plan = {p['subset']: p for p in json.loads((root/'subsets_v3.json').read_text())}
            for row in report[1:]:
                self.assertEqual(row['disguise_index'], plan[row['coalition']]['disguise_index'])
                self.assertIn('disguise_mse', row)
                self.assertIn('ground_truth_mse', row)
                self.assertEqual(row['model'], 'tiny')
                self.assertEqual(row['dataset'], 'toy')
                self.assertLess(row['disguise_residual_nnz'], row['ground_truth_residual_nnz'])
                self.assertTrue(row['residual_sparsity_pass'])
            immutable = {p: digest(root/p) for p in ['manifest.json', 'model_state_v3.pt', 'decomposition.pt', 'subsets_v3.json']}
            with patch('reconstruction.optimization.optimization_attack.run_attempt', side_effect=AssertionError('reran completed attack')):
                resumed = main(self.argv(root))
            self.assertEqual(status, resumed)
            self.assertEqual(immutable, {p: digest(root/p) for p in immutable})

    def test_decomposition_failure_saves_diagnostics_and_launches_no_attacks(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            error = ResidualSearchError('No candidate met the requested residual gap', dict(residual_nnz_gap=1))
            with patch('experiments.coalition_disguise.workflow.solve_residual', side_effect=error), \
                    patch('experiments.coalition_disguise.workflow.attack') as attack:
                with self.assertRaises(ResidualSearchError):
                    main(self.argv(root))
            attack.assert_not_called()
            self.assertFalse((root/'decomposition.pt').exists())
            saved = json.loads((root/'decomposition_failure.json').read_text())
            self.assertEqual(saved['state'], 'decomposition_failed')
            self.assertEqual(saved['diagnostics']['residual_nnz_gap'], 1)

    def test_residual_settings_are_frozen_and_inherited_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            main(self.argv(root, '--prepare-only', '--residual-nnz-gap', '2'))
            before = digest(root/'decomposition.pt')
            main(['--output', str(root), '--prepare-only'])
            self.assertEqual(before, digest(root/'decomposition.pt'))
            for options in (['--residual-nnz-gap', '3'], ['--decomposition', 'sparse'],
                            ['--max-disguise-distance-squared', '0']):
                with self.assertRaisesRegex(ValueError, 'differ'):
                    main(self.argv(root, '--prepare-only', *options))
            self.assertEqual(before, digest(root/'decomposition.pt'))

    def test_failed_baseline_launches_no_subset_attacks(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            calls = []
            def failed_attack(model, observation, label, name, args, destination):
                calls.append(destination.name)
                destination.mkdir()
                z = torch.zeros(1, 3, 2, 2)
                summary = dict(best_metrics=dict(attack_loss=1., selected_label=int(label.item())),
                    total_iterations=1, reason='no_improvement', run_metrics=dict(restarts_started=1))
                return summary, z
            with patch('experiments.coalition_disguise.workflow.attack', side_effect=failed_attack):
                status = main(self.argv(root))
            self.assertEqual(calls, ['full'])
            self.assertEqual(status[0]['state'], 'baseline_failed')
            self.assertEqual(status[0]['completed_subsets'], 0)
            self.assertEqual(len(json.loads((root/'report.json').read_text())), 1)

    def test_saved_model_and_sample_reject_incompatible_resume(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            main(self.argv(root, '--prepare-only'))
            before = digest(root/'decomposition.pt')
            with self.assertRaisesRegex(ValueError, 'differ'):
                main(self.argv(root, '--prepare-only', '--seed', '2027'))
            self.assertEqual(before, digest(root/'decomposition.pt'))
            state = torch.load(root/'model_state_v3.pt', weights_only=True)
            state['1.bias'][0] += 1
            torch.save(state, root/'model_state_v3.pt')
            with self.assertRaisesRegex(ValueError, 'differ'):
                main(self.argv(root, '--prepare-only'))

    def test_baseline_success_is_based_on_image_recovery_after_the_chosen_budget(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)/'experiment'
            status = main(self.argv(root, '--max-iterations', '1'))
            self.assertTrue(status[0]['baseline_passed'])
            self.assertEqual(status[0]['completed_subsets'], 6)
            report = json.loads((root/'report.json').read_text())
            self.assertEqual(report[0]['stopping_reason'], 'max_iterations')
            self.assertLess(report[0]['ground_truth_mse'], .01)

    def test_model_initialization_is_seeded_independently(self):
        image = self.private[0][0][None]
        a = build_model('tiny', self.data, image, 2026)
        torch.rand(100)
        b = build_model('tiny', self.data, image, 2026)
        c = build_model('tiny', self.data, image, 2027)
        self.assertEqual(model_digest(a), model_digest(b))
        self.assertNotEqual(model_digest(a), model_digest(c))


class RegisteredModelDatasetTests(unittest.TestCase):
    def test_registered_models_adapt_to_both_dataset_shapes(self):
        torch.set_num_threads(2)
        for dataset in ('cifar10', 'mnist'):
            definition = dataset_dict[dataset]
            image = torch.zeros(1, definition.input_channels, *definition.input_dimensions)
            for name in ('lenet5', 'resnet18', 'resnet50'):
                with self.subTest(dataset=dataset, model=name):
                    model = build_model(name, definition, image, 2026)
                    with torch.no_grad():
                        output = model(image)
                    self.assertEqual(output.shape, (1, definition.class_count))
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertFalse(model.training)
                    del model


if __name__ == '__main__':
    unittest.main()
