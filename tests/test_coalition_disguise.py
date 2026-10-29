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
            with patch('reconstruction.optimization.optimization_attack.run_attempt', side_effect=AssertionError('reran completed job')):
                saved, resumed = attack(model, g, label, 'geiping', self.args(2), path)
            self.assertEqual(saved, summary)
            self.assertTrue(torch.equal(result, resumed))
            torch.save(torch.zeros_like(result), path / 'reconstruction.pt')
            with self.assertRaises(ValueError):
                attack(model, g, label, 'geiping', self.args(2), path)

    def test_exhausted_search_is_not_infeasibility_claim(self):
        with self.assertRaisesRegex(RuntimeError, 'not an infeasibility proof'):
            solve(torch.zeros(2), torch.ones(1, 2), partitions=1)


class ReconstructionCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def args(self, **changes):
        values = dict(seed=2026, restarts=1, lr=None, patience=5,
                      minimum_change=1e-10, max_iterations=40,
                      diagnostics_interval=2, attack_profile='calibrated',
                      lbfgs_tolerance_grad=1e-12, lbfgs_tolerance_change=1e-15)
        values.update(changes)
        return argparse.Namespace(**values)

    def fixture(self):
        torch.manual_seed(47)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 10)).eval()
        image = .1 + .8 * torch.rand(1, 3, 2, 2)
        label = torch.tensor([2])
        return model, image, label, gradient(model, image, label)

    def test_registered_preset_profile_is_unchanged(self):
        from dataclasses import asdict
        from validation.registries.registry_builder import build_registries
        for name in ('idlg', 'dlg', 'geiping'):
            args = self.args(attack_profile='preset', restarts=None)
            actual = build_attack(name, args).description.attack
            expected = build_registries()['attacks'][name].build()
            expected.base_seed = args.seed
            self.assertEqual(asdict(actual), asdict(expected))

    def test_sigmoid_lbfgs_recovers_identifiable_image_and_matches_aggregate_control(self):
        from experiments.coalition_disguise.common import model_digest
        model, image, label, g = self.fixture()
        checksum = model_digest(model)
        # Nonzero cancelling perturbation plus ordinary float32 serialization.
        shares = torch.zeros(3, len(g))
        shares[0] = g
        shares[0, 0] += .001
        shares[1, 0] -= .001
        full = shares.sum(0)
        torch.testing.assert_close(full, g, rtol=1e-6, atol=1e-7)
        ranges = []
        handle = model.register_forward_pre_hook(lambda module, inputs: ranges.append(
            (float(inputs[0].detach().min()), float(inputs[0].detach().max()))))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                results = []
                for name, target in [('full', full), ('original', g)]:
                    summary, z = attack(model, target, label, 'idlg', self.args(), Path(tmp)/name,
                                        input_shape=(3, 2, 2))
                    self.assertEqual(summary['reason'], 'no_improvement')
                    self.assertLess(float((z - image).square().mean()), 1e-8)
                    self.assertEqual(summary['best_metrics']['outside_pixel_fraction'], 0)
                    self.assertEqual(summary['best_metrics']['selected_label'], int(label))
                    results.append(z)
                self.assertLess(float((results[0] - results[1]).square().mean()), 1e-8)
            self.assertTrue(ranges)
            self.assertTrue(all(low >= 0 and high <= 1 for low, high in ranges))
            self.assertEqual(checksum, model_digest(model))
        finally:
            handle.remove()

    def test_diagnostics_measure_physical_and_sigmoid_coordinate_gradients(self):
        from experiments.coalition_disguise.attacks import TraceObserver
        from reconstruction.optimization.labels.label_policy_def import label_state
        from reconstruction.optimization.optimization_state import optimization_state
        with tempfile.TemporaryDirectory() as tmp:
            active = TraceObserver(build_attack('idlg', self.args()).observer, Path(tmp)/'trace.csv', interval=1)
            try:
                variable = torch.tensor([[[[0., 1.]]]], requires_grad=True)
                image = variable.sigmoid()
                data = active.inspect_step(image, variable, label_state(torch.tensor([2])),
                                           image.sum(), optimization_state(1, 0))
                self.assertEqual(data['image_gradient_linf'], 1)
                self.assertAlmostEqual(data['variable_gradient_linf'], .25)
                self.assertEqual(data['outside_pixel_fraction'], 0)
            finally:
                active.close()

    def test_resume_rejects_changed_observation_and_skips_completed_optimization(self):
        from unittest.mock import patch
        model, _, label, g = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            saved, result = attack(model, g, label, 'idlg', self.args(), path, input_shape=(3, 2, 2))
            with patch('reconstruction.optimization.optimization_attack.run_attempt', side_effect=AssertionError('reran')):
                cached, z = attack(model, g, label, 'idlg', self.args(), path, input_shape=(3, 2, 2))
            self.assertEqual(saved, cached)
            self.assertTrue(torch.equal(result, z))
            with self.assertRaisesRegex(ValueError, 'Job inputs/configuration differ'):
                attack(model, g + .01, label, 'idlg', self.args(), path, input_shape=(3, 2, 2))

    def test_source_update_can_reuse_legacy_decomposition_manifest_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            old = dict(checkpoint_sha256='same', arguments={'seed': 2026},
                       source_sha256={'run.py': 'old'}, torch_version='old', python_version='old')
            claim_manifest(path, old)
            before = (path/'manifest.json').read_bytes()
            updated = dict(old, source_sha256={'run.py': 'new'}, torch_version='new')
            claim_manifest(path, updated)
            self.assertEqual(before, (path/'manifest.json').read_bytes())
            with self.assertRaises(ValueError):
                claim_manifest(path, dict(updated, checkpoint_sha256='different'))

    def test_calibration_requires_both_controls_and_all_stalled_restarts(self):
        from copy import deepcopy
        from experiments.coalition_disguise.diagnostics import calibration_diagnosis
        checks = dict(decomposition=dict(all_coalitions_pass=True, float32_aggregation_pass=True,
                                        float64_aggregation_pass=True), checkpoint_data_gradient_equal=True,
                      flatten_unpack_roundtrip_equal=True, full_idlg_label_correct=True)
        passed = dict(preset='idlg', evaluation=dict(private_mse=.001, outside_pixel_fraction=0),
                      attempt_stopping_reasons=['no_improvement', 'no_improvement'])
        failed = deepcopy(passed)
        failed['evaluation']['private_mse'] = .6
        self.assertEqual(calibration_diagnosis(checks, passed, passed, 2)['stage'], 'ready_for_coalitions')
        self.assertEqual(calibration_diagnosis(checks, failed, failed, 2)['stage'], 'reconstruction_calibration')
        self.assertEqual(calibration_diagnosis(checks, failed, passed, 2)['stage'], 'aggregation_sensitivity_or_optimization_basin')
        for reason in ('max_iterations',):
            capped = deepcopy(passed)
            capped['attempt_stopping_reasons'][0] = reason
            self.assertFalse(calibration_diagnosis(checks, capped, passed, 2)['calibrated'])
        self.assertFalse(calibration_diagnosis(checks, passed, passed, 3)['calibrated'])
        out_of_bounds = deepcopy(passed)
        out_of_bounds['evaluation']['outside_pixel_fraction'] = .1
        self.assertFalse(calibration_diagnosis(checks, out_of_bounds, passed, 2)['calibrated'])

    def test_report_retains_previous_configurations(self):
        from experiments.coalition_disguise.run import refresh_report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for config, mse in [('idlg_old', .6), ('idlg_new', .001)]:
                folder = root/config/'full'
                folder.mkdir(parents=True)
                (folder/'summary.json').write_text(json.dumps(dict(
                    evaluation=dict(private_mse=mse), best_metrics=dict(attack_loss=.01),
                    total_iterations=5, reason='no_improvement', run_metrics=dict(restarts_started=1))))
            records = refresh_report(root)
            self.assertEqual(len(records), 2)
            self.assertEqual({r['private_mse'] for r in records}, {.6, .001})

    def test_evaluation_uses_selected_soft_labels_for_attack_residual(self):
        from experiments.coalition_disguise.run import evaluate
        model, image, label, g = self.fixture()
        probabilities = torch.ones(1, 10)/10
        observation = gradient(model, image, probabilities)
        result = evaluate(model, image, image, image, label, observation, g, [0],
                          dict(label_probabilities=probabilities.tolist()))
        self.assertEqual(result['gradient_l2_squared'], 0)
        self.assertGreater(result['true_label_gradient_l2_squared'], .01)
        self.assertEqual(result['private_mse'], 0)

    def test_legacy_diagnostics_inspect_saved_out_of_range_image_without_modifying_it(self):
        from experiments.coalition_disguise.diagnostics import inspect_saved_runs
        from experiments.coalition_disguise.common import digest
        model, image, label, g = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'idlg_old'/'full'
            folder.mkdir(parents=True)
            torch.save(image+2, folder/'reconstruction.pt')
            (folder/'summary.json').write_text(json.dumps(dict(best_metrics=dict(attack_loss=1))))
            (folder/'trajectory.csv').write_text('step,restart,attack_loss\n1,0,1\n2,0,1\n3,0,1\n')
            hashes = {p.name: digest(p) for p in folder.iterdir()}
            artifact = dict(g=g, shares=torch.stack([g, g*0, g*0]))
            reports = inspect_saved_runs(root, model, image, label, artifact)
            self.assertEqual(reports[0]['outside_pixel_fraction'], 1)
            self.assertEqual(reports[0]['identical_final_objective_rows'], 3)
            self.assertLess(reports[0]['clamped_private_mse'], reports[0]['private_mse'])
            self.assertEqual(hashes, {p.name: digest(p) for p in folder.iterdir()})

    def test_invalid_stop_settings_are_rejected(self):
        for changes in [dict(max_iterations=0), dict(minimum_change=0), dict(patience=0)]:
            with self.assertRaises(ValueError):
                build_attack('idlg', self.args(**changes))


if __name__ == '__main__':
    unittest.main()
