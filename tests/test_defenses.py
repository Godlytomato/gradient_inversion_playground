"""Regression checks for the numerical defense protocol and public presets."""

import contextlib
import io
import random
import unittest
import warnings

import torch

from run_def import function_package, grad_transform
from run_utils.grad_prod import grad_prod
from run_utils.transformer import transformer
from transformations.draft_implimentations.all import all_draft
from transformations.draft_implimentations.highest_and_lowest import highest_lowest
from transformations.draft_implimentations.sensitivity_slices import sensitivity_slices
from transformations.ranking_implimentations.graddefense_sensitivity import graddefense_sensitivity
from transformations.transform_def import grad_package, transformation_context
from transformations.transformation_implimentations.quantization import quantization_delta
from transformations.transformation_implimentations.threasholds.relative_gradient_scale import relative_gradient_scale
from validation.registries.registry_builder import build_registries


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
        self.classifier = torch.nn.Linear(3, 2)

    def forward(self, images):
        return self.classifier(torch.tanh(self.fc1(images.flatten(1))))


class DefenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.presets = build_registries()['defenses']

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyClassifier().double()
        self.image = torch.tensor([[[.2, .4, .8]]], dtype=torch.float64)
        self.batch = [(self.image, 1)]
        self.grads = grad_prod(self.model, self.batch)
        self.context = transformation_context(model=self.model, batch=self.batch)

    def execute(self, description):
        return transformer([gradient.clone() for gradient in self.grads], description, self.context)

    def test_soteria_changes_only_complete_features_of_auto_classifier(self):
        result = self.execute(self.presets['soteria'].build(percent=.5))
        names = [name for name, _ in self.model.named_parameters()]
        for name, before, after, mask in zip(names, self.grads, result.grads, result.missing_mask):
            if name == 'classifier.weight':
                self.assertEqual(int(mask.sum()), 2)
                self.assertTrue(torch.equal(mask[0], mask[1]))
                self.assertEqual(int(mask[0].sum()), 1)
            else:
                self.assertFalse(bool(mask.any()))
                torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_soteria_explicit_layer_and_feature_fraction_endpoints(self):
        for percent, expected_features in [(0., 0), (.5, 1), (1., 3)]:
            with self.subTest(percent=percent):
                result = self.execute(self.presets['soteria'].build(layer_name='fc1', percent=percent))
                self.assertEqual(int(result.missing_mask[0].sum()), 3 * expected_features)
                self.assertTrue(torch.equal(result.missing_mask[0][0], result.missing_mask[0][1]))
                self.assertEqual(sum(int(mask.sum()) for mask in result.missing_mask[1:]), 0)

    def test_historical_soteria_global_draft_fails_with_migration_instruction(self):
        description = self.presets['soteria'].build(percent=.5)
        description[0].draft_policy = function_package(name='highest_draft', parameters={'percent': .5})
        with self.assertRaisesRegex(ValueError, 'feature_columns'):
            self.execute(description)

    def test_frobenius_sensitivity_matches_independent_finite_differences(self):
        # Central differences of ordinary parameter gradients give an
        # independent explicit Jacobian for this tiny smooth classifier.
        columns = []
        step = 1e-5
        for coordinate in range(self.image.numel()):
            plus, minus = self.image.clone(), self.image.clone()
            plus.flatten()[coordinate] += step
            minus.flatten()[coordinate] -= step
            gp = grad_prod(self.model, [(plus, 1)])
            gm = grad_prod(self.model, [(minus, 1)])
            columns.append(torch.cat([(a-b).flatten() for a, b in zip(gp, gm)]) / (2*step))
        jacobian = torch.stack(columns, dim=1)
        rng_before = torch.random.get_rng_state().clone()
        estimated = graddefense_sensitivity(grad_package(self.grads), self.context, samples=1024, seed=11)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        offset = 0
        for gradient, score in zip(self.grads, estimated):
            exact = jacobian[offset:offset+gradient.numel()].norm()
            self.assertGreater(float(exact), 0)
            self.assertLess(abs(float(score.flatten()[0]/exact)-1), .12)
            offset += gradient.numel()
        legacy = graddefense_sensitivity(grad_package(self.grads), self.context, method='legacy_sum')
        self.assertLess(float(legacy[-1].abs().max()), 1e-14)
        self.assertGreater(float(estimated[-1].flatten()[0]), .001)

    def test_skewed_slices_are_nonempty_disjoint_and_cover_atoms(self):
        for scores in [[torch.tensor([100.]), torch.zeros(9)], [torch.zeros(1), torch.zeros(21)]]:
            masks_by_slice = []
            for index in range(10):
                seed = next(seed for seed in range(1000) if random.Random(seed).sample(range(10), 1) == [index])
                masks = sensitivity_slices(scores, 10, 1, seed=seed)
                flattened = torch.cat([mask.flatten().to(torch.int64) for mask in masks])
                self.assertGreater(int(flattened.sum()), 0)
                masks_by_slice.append(flattened)
            torch.testing.assert_close(torch.stack(masks_by_slice).sum(dim=0), torch.ones_like(masks_by_slice[0]))

    def test_narrow_classifier_graddefense_applies_requested_noise(self):
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, 1), torch.nn.Tanh(), torch.nn.Linear(1, 10)).double()
        batch = [(torch.tensor([[[.4]]], dtype=torch.float64), 3)]
        original = grad_prod(model, batch)
        result = transformer([g.clone() for g in original], self.presets['graddefense'].build(relative_scale=.5),
                             transformation_context(model=model, batch=batch))
        ratio = torch.cat([(new-old).flatten() for new, old in zip(result.grads, original)]).norm() / torch.cat([g.flatten() for g in original]).norm()
        self.assertAlmostEqual(float(ratio), .5, places=12)

    def test_relative_budget_uses_selected_noise_in_global_and_layer_modes(self):
        package = grad_package([torch.ones(4, dtype=torch.float64), torch.ones(2, dtype=torch.float64)])
        masks = [torch.tensor([True, False, False, False]), torch.tensor([False, False])]
        for mode, expected_norm in [('global', .1 * 6**.5), ('layer', .2)]:
            with self.subTest(mode=mode):
                scaled = relative_gradient_scale(package, [torch.ones_like(g) for g in package.grads], masks,
                                                 self.context, relative_scale=.1, mode=mode)
                self.assertAlmostEqual(float(torch.cat([n.flatten() for n in scaled]).norm()), expected_norm, places=12)
                for noise, mask in zip(scaled, masks):
                    self.assertFalse(bool(noise[~mask].any()))

    def test_reference_targets_and_sensitivity_method_are_saved(self):
        for name in ['gaussian_noise_reference', 'uniform_noise_reference', 'laplace_noise_reference',
                     'min_svd_noise_reference', 'graddefense_reference']:
            for target in [.005, .1]:
                with self.subTest(name=name, target=target):
                    desc = self.presets[name].build(target_error=target)[0]
                    self.assertEqual(desc.transform.parameters['scaling']['parameters']['target_error'], target)
        for name in ['graddefense', 'graddefense_reference']:
            params = self.presets[name].build()[0].ranking_policy.parameters
            self.assertEqual(params['method'], 'frobenius_rademacher')
            self.assertGreater(params['samples'], 0)

    def test_solver_settings_are_forwarded_to_noise_construction(self):
        for name in ['min_svd_noise', 'min_svd_lower_bound', 'min_svd_noise_reference']:
            desc = self.presets[name].build(iterations=13, solve_iterations=17, damping=.002)[0]
            self.assertEqual(desc.ranking_policy.parameters, desc.transform.parameters['construction']['parameters'])
        for name in ['max_svd_noise', 'max_svd_lower_bound']:
            desc = self.presets[name].build(iterations=13)[0]
            self.assertEqual(desc.transform.parameters['construction']['parameters']['iterations'], 13)

    def test_default_maximum_svd_presets_enable_checked_adaptive_budget(self):
        # The public presets must not pin the core adaptive solver to ten steps.
        for name in ['max_svd_noise', 'max_svd_lower_bound', 'max_svd_pruning']:
            with self.subTest(name=name):
                desc = self.presets[name].build()[0]
                self.assertIsNone(desc.ranking_policy.parameters['iterations'])
                if desc.transform.name == 'noise':
                    self.assertIsNone(desc.transform.parameters['construction']['parameters']['iterations'])
                explicit = self.presets[name].build(iterations=13)[0]
                self.assertEqual(explicit.ranking_policy.parameters['iterations'], 13)

    def test_all_selector_returns_masks_and_executes(self):
        masks = all_draft([torch.tensor([0., -2., float('nan')])])
        self.assertEqual(masks[0].dtype, torch.bool)
        self.assertTrue(bool(masks[0].all()))
        result = self.execute([grad_transform(name='all', ranking_policy=function_package(name='in_place'),
                              draft_policy=function_package(name='all'), transform=function_package(name='zero'))])
        self.assertTrue(all(bool((g == 0).all()) for g in result.grads))

    def test_invalid_one_bit_quantization_fails_before_execution(self):
        with self.assertRaises(ValueError):
            self.presets['quantization'].build(bits=1)
        with self.assertRaises(ValueError):
            quantization_delta(grad_package([torch.zeros(3)]), [torch.ones(3, dtype=torch.bool)], self.context, bits=1)

    def test_dgp_odd_count_is_preserved(self):
        for size, percent in [(10, .1), (10, .3), (5, 1.)]:
            self.assertEqual(int(highest_lowest([torch.arange(size)], percent)[0].sum()), int(size*percent))

    def test_all_discovered_presets_execute_on_synthetic_data(self):
        self.assertIn('graddefense_reference', self.presets)
        for name, preset in self.presets.items():
            with self.subTest(name=name), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = self.execute(preset.build())
                self.assertTrue(all(bool(g.isfinite().all()) for g in result.grads))


if __name__ == '__main__':
    unittest.main()
