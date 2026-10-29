import argparse
import unittest

import torch
import torch.nn.functional as F

from experiments.coalition_disguise.attacks import build_attack
from reconstruction.optimization.initialization.implementations.uniform import (
    smooth_uniform_initialization, smooth_uniform_sigmoid_initialization,
    uniform_initialization, uniform_sigmoid_initialization,
)


class SmoothInitializationTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(seed=2026, restarts=1, lr=None, patience=200, minimum_change=1e-10,
                      max_iterations=2, attack_profile='calibrated')
        values.update(changes)
        return argparse.Namespace(**values)

    def test_seeded_smooth_image_is_the_resized_random_grid(self):
        torch.manual_seed(2026)
        actual = smooth_uniform_initialization(1, (3, 32, 32), 'cpu')
        torch.manual_seed(2026)
        expected = F.interpolate(.05+.9*torch.rand(1, 3, 4, 4), size=(32, 32),
                                 mode='bilinear', align_corners=False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertGreaterEqual(float(actual.min()), .05)
        self.assertLessEqual(float(actual.max()), .95)
        torch.manual_seed(2027)
        different = smooth_uniform_initialization(1, (3, 32, 32), 'cpu')
        self.assertFalse(torch.equal(actual, different))

    def test_initial_smoothing_keeps_every_pixel_trainable(self):
        for shape in ((3, 32, 32), (1, 28, 28), (3, 2, 2)):
            state = smooth_uniform_sigmoid_initialization(1, shape, 'cpu')
            self.assertEqual(tuple(state.variable.shape), (1, *shape))
            self.assertTrue(state.variable.is_leaf)
            self.assertTrue(state.variable.requires_grad)
            image = state.decode(state.variable)
            image.sum().backward()
            self.assertTrue((state.variable.grad > 0).all())

    def test_smooth_initialization_removes_initial_pixel_noise(self):
        torch.manual_seed(2026)
        smooth = smooth_uniform_initialization(1, (3, 32, 32), 'cpu')
        torch.manual_seed(2026)
        noisy = uniform_initialization(1, (3, 32, 32), 'cpu')
        variation = lambda z: ((z[..., 1:] - z[..., :-1]).abs().mean()
                               + (z[..., 1:, :] - z[..., :-1, :]).abs().mean())
        self.assertLess(float(variation(smooth)), float(variation(noisy)) * .25)
        # Small identifiable fixtures retain their old full-resolution start.
        torch.manual_seed(2026)
        small = smooth_uniform_sigmoid_initialization(1, (3, 2, 2), 'cpu')
        torch.manual_seed(2026)
        original = uniform_sigmoid_initialization(1, (3, 2, 2), 'cpu')
        self.assertTrue(torch.equal(small.variable, original.variable))

    def test_calibrated_default_and_previous_initializer_override_are_recorded(self):
        for name in ('idlg', 'dlg'):
            default = build_attack(name, self.args()).description.attack
            previous = build_attack(name, self.args(initialization='uniform_sigmoid')).description.attack
            self.assertEqual(default.initialization_policy, 'smooth_uniform_sigmoid')
            self.assertEqual(previous.initialization_policy, 'uniform_sigmoid')
        preset = build_attack('idlg', self.args(attack_profile='preset')).description.attack
        self.assertEqual(preset.initialization_policy, 'random')

    def test_incompatible_parameterization_is_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'Sigmoid initialization'):
            build_attack('geiping', self.args(initialization='smooth_uniform_sigmoid'))


if __name__ == '__main__':
    unittest.main()
