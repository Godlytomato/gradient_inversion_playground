"""Independent analytic checks for the residual-aware directional theorem."""
import math
import unittest

import torch

from transformations.reconstruction_bound import residual_aware_bound, required_noise_l2


class ResidualBoundTests(unittest.TestCase):
    def test_quadratic_ray_with_nonzero_residual_is_tight(self):
        error, gain, mu, residual = .7, 2., 3., .4
        noise = gain * error + .5 * mu * error ** 2 + residual
        self.assertAlmostEqual(residual_aware_bound(noise, residual, gain, mu)["lower_bound_l2"], error, places=14)

    def test_affine_minimum_and_maximum_directions_are_both_tight(self):
        self.assertEqual(residual_aware_bound(2., 0., 10., 0.)["lower_bound_l2"], .2)
        self.assertEqual(residual_aware_bound(2., 0., 1., 0.)["lower_bound_l2"], 2.)

    def test_calibration_explicitly_allows_relative_and_absolute_residual(self):
        radius, gain, mu, absolute, fraction = 1.7, .4, 2., .3, .1
        required = required_noise_l2(radius, gain, mu, absolute, fraction)
        self.assertAlmostEqual(required, (absolute + gain*radius + .5*mu*radius**2)/(1-fraction))
        actual = residual_aware_bound(required, absolute+fraction*required, gain, mu)
        self.assertAlmostEqual(actual["lower_bound_l2"], radius, places=14)

    def test_stalled_reconstruction_can_have_zero_bound(self):
        self.assertEqual(residual_aware_bound(2., 2., 1., 1.)["lower_bound_l2"], 0.)
        self.assertEqual(residual_aware_bound(2., 3., 1., 1.)["status"], "uninformative")

    def test_previous_exact_fit_example_uses_its_real_positive_residual(self):
        amplitude = 7.396879308487042
        observed = 10-amplitude
        residual = 9.995-observed
        result = residual_aware_bound(amplitude, residual, .1, 1.)
        self.assertLess(result["lower_bound_l2"] ** 2, .01)
        self.assertAlmostEqual(result["lower_bound_l2"], math.sqrt(.02)-.1, places=12)

    def test_degenerate_and_extreme_values(self):
        self.assertEqual(residual_aware_bound(1., 0., 0., 0.)["status"], "incompatible_residual_assumption")
        self.assertIsNone(required_noise_l2(1., 0., 0.))
        self.assertEqual(required_noise_l2(0., 0., 0., 10.), 0.)
        self.assertAlmostEqual(residual_aware_bound(1e308, 0., 1., 1e308)["lower_bound_l2"], math.sqrt(2), places=14)
        small = residual_aware_bound(1e-308, 0., 0., 1e308)["lower_bound_l2"]
        self.assertAlmostEqual(small/1e-308, math.sqrt(2), places=14)
        self.assertEqual(residual_aware_bound(1e308, 0., 1e-308, 0.)["status"], "outside_numeric_range")
        self.assertTrue(math.isfinite(required_noise_l2(1e308, 0., 5e-324)))
        self.assertGreater(required_noise_l2(1e308, 0., 5e-324), 1e292)

    def test_invalid_inputs_are_rejected(self):
        for value in (-1., float('nan'), float('inf'), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                residual_aware_bound(value, 0., 1., 1.)
        for fraction in (-.1, 1., 2., float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                required_noise_l2(1., 1., 1., residual_fraction=fraction)

    def test_vector_quadratics_with_known_uniform_constant(self):
        rng = torch.Generator().manual_seed(194)
        for _ in range(30):
            matrix = torch.randn(5, 3, dtype=torch.float64, generator=rng)
            curvature = torch.randn(5, dtype=torch.float64, generator=rng)
            displacement = .3 * torch.randn(3, dtype=torch.float64, generator=rng)
            difference = matrix @ displacement + .5 * displacement.square().sum() * curvature
            eta = difference + .05 * torch.randn(5, dtype=torch.float64, generator=rng)
            amplitude = float(eta.norm())
            residual = float((difference-eta).norm())
            gain = float((matrix.T @ (eta/amplitude)).norm())
            result = residual_aware_bound(amplitude, residual, gain, float(curvature.norm()))
            self.assertLessEqual(result["lower_bound_l2"], float(displacement.norm())+1e-12)


if __name__ == '__main__':
    unittest.main()
