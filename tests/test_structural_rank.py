"""Structural certificates must depend on the actual recognized architecture."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.lenet import LeNet5, change_channels
from transformations.structural_rank import gradient_jacobian_rank_bound


class StructuralRankTests(unittest.TestCase):
    def context(self, channels=3, model=None, loss_function=None):
        if model is None:
            model = LeNet5()
            change_channels(model, channels)
        return SimpleNamespace(model=model, loss_function=loss_function)

    def certificate(self, context, batch=1, channels=3, size=32):
        return gradient_jacobian_rank_bound(
            context, torch.zeros(batch, channels, size, size))

    def test_rgb_lenet_has_proven_input_nullspace(self):
        result = self.certificate(self.context())
        self.assertEqual(result["input_dimension"], 3072)
        self.assertEqual(result["bottleneck_dimension"], 1536)
        self.assertEqual(result["early_parameter_count"], 456)
        self.assertEqual(result["rank_upper_bound"], 1992)
        self.assertEqual(result["nullity_lower_bound"], 1080)
        self.assertIs(result["certified"], True)

    def test_batch_shares_first_layer_gradient_coordinates(self):
        result = self.certificate(self.context(), batch=2)
        self.assertEqual(result["input_dimension"], 6144)
        self.assertEqual(result["rank_upper_bound"], 2 * 1536 + 456)
        self.assertEqual(result["nullity_lower_bound"], 2616)

    def test_mnist_dimensions_do_not_force_nullspace(self):
        self.assertIsNone(self.certificate(self.context(1), channels=1, size=28))

    def test_standard_loss_supported_but_custom_loss_is_not(self):
        self.assertIsNotNone(self.certificate(self.context(loss_function=nn.CrossEntropyLoss())))
        self.assertIsNone(self.certificate(self.context(loss_function=lambda output, labels: output.sum())))
        loss = nn.CrossEntropyLoss()
        loss.forward = lambda output, labels: output.sum()
        self.assertIsNone(self.certificate(self.context(loss_function=loss)))

    def test_unknown_subclass_and_replaced_forwards_are_not_certified(self):
        class CustomLeNet(LeNet5):
            pass
        self.assertIsNone(self.certificate(self.context(model=nn.Linear(3, 3))))
        self.assertIsNone(self.certificate(self.context(model=CustomLeNet())))
        for name in (None, "conv1", "pool", "fc1"):
            with self.subTest(module=name):
                context = self.context()
                child = context.model if name is None else getattr(context.model, name)
                child.forward = lambda images: images
                self.assertIsNone(self.certificate(context))
        context = self.context()
        with patch.object(LeNet5, "forward", lambda model, images: images):
            self.assertIsNone(self.certificate(context))

    def test_inspection_does_not_initialize_lazy_weights(self):
        context = self.context()
        lazy = context.model.fc1
        before = torch.random.get_rng_state().clone()
        self.assertTrue(lazy.has_uninitialized_params())
        self.assertIsNotNone(self.certificate(context))
        self.assertTrue(lazy.has_uninitialized_params())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_hooks_disable_certificate(self):
        for name in (None, "conv1", "fc1"):
            context = self.context()
            child = context.model if name is None else getattr(context.model, name)
            for registration in ("register_forward_hook", "register_forward_pre_hook",
                                 "register_full_backward_hook"):
                with self.subTest(module=name, hook=registration):
                    handle = getattr(child, registration)(lambda *args: None)
                    try:
                        self.assertIsNone(self.certificate(context))
                    finally:
                        handle.remove()
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *args: None)
        try:
            self.assertIsNone(self.certificate(self.context()))
        finally:
            handle.remove()
        context = self.context()
        handle = context.model.conv1.weight.register_hook(lambda gradient: gradient)
        try:
            self.assertIsNone(self.certificate(context))
        finally:
            handle.remove()

    def test_altered_geometry_parameter_sharing_and_added_parameters_bypass(self):
        for alteration in (
            lambda model: setattr(model.conv1, "stride", (2, 2)),
            lambda model: setattr(model.pool, "ceil_mode", True),
            lambda model: setattr(model.pool, "padding", 1),
            lambda model: setattr(model.fc2, "weight", model.fc3.weight),
            lambda model: model.register_parameter("extra", nn.Parameter(torch.ones(1))),
            lambda model: model.register_buffer("extra", torch.ones(1)),
        ):
            context = self.context()
            alteration(context.model)
            self.assertIsNone(self.certificate(context))

    def test_inspection_preserves_rng_parameters_gradients_modes_and_dtypes(self):
        context = self.context()
        model = context.model.double().train()
        images = torch.zeros(1, 3, 32, 32, dtype=torch.float64)
        model(images)  # Initialize lazy weights before taking the snapshot.
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = [(parameter.detach().clone(), parameter.grad.clone(), parameter.dtype)
                  for parameter in model.parameters()]
        rng_state = torch.random.get_rng_state().clone()
        self.assertIsNotNone(gradient_jacobian_rank_bound(context, images))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertTrue(model.training)
        self.assertEqual(images.dtype, torch.float64)
        for parameter, (value, gradient, dtype) in zip(model.parameters(), before):
            self.assertTrue(torch.equal(parameter, value))
            self.assertTrue(torch.equal(parameter.grad, gradient))
            self.assertEqual(parameter.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
