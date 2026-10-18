"""Measured residuals and actual noise directions control post-attack bounds."""
from types import SimpleNamespace
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch

from run_utils.bound_evaluation import evaluate_reconstruction_bounds
from transformations.transform_def import grad_package
from transformations.ranking_implimentations.transform_utils import model_gradient_vector


class GradientMap(torch.nn.Module):
    def __init__(self, quadratic=False):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(1 if quadratic else 2, dtype=torch.float64))
        self.quadratic = quadratic

    def forward(self, x):
        features = x.flatten(1).square() if self.quadratic else x.flatten(1) * torch.tensor([10., 1.])
        signal = 2 * (features * self.theta).sum(1)
        return torch.stack((signal, torch.zeros_like(signal)), 1)


class BoundEvaluationTests(unittest.TestCase):
    def context(self, truth, eta, quadratic=False):
        model = GradientMap(quadratic)
        labels = torch.ones(len(truth), dtype=torch.long)
        original = model_gradient_vector(model, truth, labels, create_graph=False).detach()
        calibration = dict(bound_method='directional_residual_v1', mu_j=0.,
                           residual_l2=0., residual_fraction=.1, direction_effect=10.)
        return SimpleNamespace(
            model=model, batch=list(zip(truth, labels)), original_gradients=[original],
            gradients=grad_package([original+eta]),
            experiment_metadata={'numerical_diagnostics': {'lower_bound': calibration}},
        )

    def test_reference_uses_actual_minimum_direction_and_measured_residual(self):
        truth = torch.tensor([[.25, .25]], dtype=torch.float64)
        ctx = self.context(truth, torch.tensor([0., .2], dtype=torch.float64))
        reconstructed = truth + torch.tensor([[0., .2]])
        result = evaluate_reconstruction_bounds(ctx, {'practical': reconstructed})['practical']
        self.assertEqual(result['direction_effect'], 1.)
        self.assertAlmostEqual(result['gradient_residual_l2'], 0., places=7)
        self.assertAlmostEqual(result['conditional_lower_bound_mse'], .02, places=7)
        self.assertTrue(result['calibration_residual_condition_met'])
        self.assertFalse(result['certified'])

    def test_failed_attack_gets_zero_bound_without_being_an_error(self):
        truth = torch.tensor([[.25, .25]], dtype=torch.float64)
        ctx = self.context(truth, torch.tensor([0., .2], dtype=torch.float64))
        result = evaluate_reconstruction_bounds(ctx, {'practical': truth})['practical']
        self.assertEqual(result['conditional_lower_bound_mse'], 0.)
        self.assertFalse(result['calibration_residual_condition_met'])

    def test_bad_smoothness_assumption_is_reported_instead_of_false_bound(self):
        truth = torch.tensor([[1.]], dtype=torch.float64)
        ctx = self.context(truth, torch.tensor([3.], dtype=torch.float64), quadratic=True)
        result = evaluate_reconstruction_bounds(ctx, {'practical': torch.tensor([[2.]], dtype=torch.float64)})['practical']
        self.assertEqual(result['status'], 'assumptions_contradicted')
        self.assertIsNone(result['conditional_lower_bound_mse'])
        self.assertEqual(result['candidate_lower_bound_mse'], 2.25)

    def test_assignment_matches_evaluation_labels_and_error(self):
        truth = torch.tensor([[[[.1, .1]]], [[[.4, .4]]]], dtype=torch.float64)
        ctx = self.context(truth, torch.tensor([0., .1], dtype=torch.float64))
        reconstructed = (truth + torch.tensor([[[[0., .1]]]], dtype=torch.float64)).flip(0)
        result = evaluate_reconstruction_bounds(ctx, {'practical': reconstructed})['practical']
        self.assertEqual(result['assignment'], [1, 0])
        self.assertAlmostEqual(result['conditional_lower_bound_mse'], .005)

    def test_unavailable_diagnostics_do_not_discard_attack_results(self):
        truth = torch.tensor([[.25, .25]], dtype=torch.float64)
        ctx = self.context(truth, torch.tensor([0., .2], dtype=torch.float64))
        ctx.original_gradients = None
        self.assertEqual(evaluate_reconstruction_bounds(ctx, {'practical': truth})['status'], 'unavailable')
        ctx.experiment_metadata = {}
        self.assertEqual(evaluate_reconstruction_bounds(ctx, {'practical': truth}), {})

    def test_actual_runner_persists_measured_residual_bound(self):
        from torch.utils.data import TensorDataset
        from data.dataset_def import dataset_def
        from data.dataset_registry import dataset_dict
        from database import connection
        from database.setup import setup_database
        from validation.generator import generate_run
        from validation.registries.registry_builder import build_registries
        import run as runner
        torch.set_num_threads(1)
        registry = build_registries()
        description = generate_run(registry['attacks']['dlg'], registry['defenses']['gaussian_noise_reference'],
                                   registry['models']['lenet5'], registry['datasets']['cifar10'], .02)
        description.gradient_transformations[0].transform.parameters['scaling']['name'] = 'lower_bound_threshold'
        description.observer.fail_policies[0].parameters['limit'] = 1
        description.attack.optimizer_description.parameters['max_iter'] = 1
        fixture = dataset_def('cifar10', lambda: TensorDataset(
            torch.rand(8, 3, 32, 32, generator=torch.Generator().manual_seed(123)), torch.arange(8)),
            10, 3, (32, 32))
        with tempfile.TemporaryDirectory() as directory, patch.object(connection, 'DB_PATH', Path(directory)/'runs.db'), patch.dict(dataset_dict, {'cifar10': fixture}):
            setup_database()
            runner.execute_description(description)
            with sqlite3.connect(connection.DB_PATH) as database:
                summary = json.loads(database.execute('SELECT summary FROM runs').fetchone()[0])
            bounds = summary['experiment_metadata']['reconstruction_bounds']
            self.assertIn('gradient_residual_l2', bounds['practical'])
            self.assertEqual(bounds['practical']['bound_method'], 'directional_residual_v1')
            self.assertFalse(bounds['practical']['certified'])


if __name__ == '__main__':
    unittest.main()
