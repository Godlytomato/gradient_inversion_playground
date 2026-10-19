"""Operator identity and ownership contracts for verified-triplet reuse."""
import unittest

import torch

from transformations.svd_cache import lookup_singular_triplet, store_singular_triplet
from transformations.transform_def import transformation_context


class CacheModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
        self.register_buffer("offset", torch.zeros(2, dtype=torch.float64))
        self.child = torch.nn.Identity()

    def forward(self, value):
        return value * self.weight + self.offset


class SumLossModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(2))
        self.reduction = "sum"

    def forward(self, output, labels):
        weighted = output * self.weight
        return weighted.sum() if self.reduction == "sum" else weighted.mean()


class SingularTripletCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = CacheModel().eval()
        self.images = torch.tensor([[.2, .4]], dtype=torch.float64)
        self.labels = torch.tensor([1])
        self.context = transformation_context(self.model, [(self.images[0], 1)])
        self.settings = {"solver": "checked_test", "iterations": 17, "seed": 3}
        self.triplet = (
            torch.tensor(2., requires_grad=True),
            torch.tensor([1., 0.], requires_grad=True),
            torch.tensor([0., 1.], requires_grad=True),
        )
        self.diagnostic = {"status": "converged_estimate", "history": [{"residual": 1e-6}]}

    def store(self, which="min", settings=None):
        store_singular_triplet(
            self.context, which, self.settings if settings is None else settings,
            self.images, self.labels, self.triplet, self.diagnostic,
        )

    def lookup(self, which="min", settings=None):
        return lookup_singular_triplet(
            self.context, which, self.settings if settings is None else settings,
            self.images, self.labels,
        )

    def test_fresh_tensors_with_same_values_hit_and_restore_diagnostic(self):
        self.store()
        self.images = self.images.clone()
        self.labels = self.labels.clone()
        self.context.numerical_diagnostics["min_svd"] = {"status": "other"}
        result = self.lookup()
        self.assertIsNotNone(result)
        for actual, expected in zip(result, self.triplet):
            torch.testing.assert_close(actual, expected)
            self.assertFalse(actual.requires_grad)
        self.assertEqual(self.context.numerical_diagnostics["min_svd"], self.diagnostic)

    def test_stored_and_returned_tensors_and_diagnostics_are_isolated(self):
        self.store()
        with torch.no_grad():
            self.triplet[1].fill_(7.)
        self.diagnostic["history"][0]["residual"] = 99.
        first = self.lookup()
        first[1].fill_(8.)
        self.context.numerical_diagnostics["min_svd"]["history"][0]["residual"] = 100.
        second = self.lookup()
        torch.testing.assert_close(second[1], torch.tensor([1., 0.]))
        self.assertEqual(self.context.numerical_diagnostics["min_svd"]["history"][0]["residual"], 1e-6)

    def test_input_values_labels_shapes_and_dtypes_invalidate(self):
        for change in (
            lambda: self.images.add_(.1),
            lambda: self.labels.add_(1),
            lambda: setattr(self, "images", self.images.flatten()),
            lambda: setattr(self, "images", self.images.float()),
        ):
            with self.subTest(change=change):
                self.store()
                change()
                self.assertIsNone(self.lookup())

    def test_settings_change_invalidates_but_mapping_order_does_not(self):
        self.store()
        self.assertIsNotNone(self.lookup(settings=dict(reversed(list(self.settings.items())))))
        self.assertIsNone(self.lookup(settings={**self.settings, "iterations": 18}))

    def test_model_parameter_versions_and_replacements_invalidate(self):
        self.store()
        with torch.no_grad():
            self.model.weight.add_(1.)
        self.assertIsNone(self.lookup())
        self.store()
        self.model.weight = torch.nn.Parameter(self.model.weight.detach().clone())
        self.assertIsNone(self.lookup())

    def test_buffers_and_loss_identity_invalidate(self):
        self.store()
        self.model.offset.add_(1.)
        self.assertIsNone(self.lookup())
        self.store()
        self.context.loss_function = lambda output, labels: output.sum()
        self.assertIsNone(self.lookup())

    def test_loss_buffer_versions_and_reduction_invalidate(self):
        self.context.loss_function = SumLossModule()
        self.store()
        self.assertIsNotNone(self.lookup())
        self.context.loss_function.weight.add_(1.)
        self.assertIsNone(self.lookup())
        self.store()
        self.context.loss_function.reduction = "mean"
        self.assertIsNone(self.lookup())

    def test_loss_parameters_and_training_flag_invalidate(self):
        self.context.loss_function = SumLossModule()
        self.context.loss_function.scale = torch.nn.Parameter(torch.tensor(1.))
        self.store()
        with torch.no_grad():
            self.context.loss_function.scale.add_(1.)
        self.assertIsNone(self.lookup())
        self.store()
        self.context.loss_function.eval()
        self.assertIsNone(self.lookup())

    def test_arbitrary_callable_loss_is_never_cached(self):
        state = {"scale": 1.}
        self.context.loss_function = lambda output, labels: state["scale"] * output.sum()
        self.store()
        self.assertIsNone(self.lookup())
        state["scale"] = 2.
        self.assertIsNone(self.lookup())

    def test_model_hyperparameters_and_nested_immutable_configuration_invalidate(self):
        self.model.bn = torch.nn.BatchNorm1d(2).eval()
        self.model.conv = torch.nn.Conv1d(2, 2, 1).eval()
        self.model.configuration = ("version", (1, True, None))
        self.store()
        self.assertIsNotNone(self.lookup())
        self.model.bn.eps = .01
        self.assertIsNone(self.lookup())
        self.store()
        self.model.conv.stride = (2,)
        self.assertIsNone(self.lookup())
        self.store()
        self.model.configuration = ("version", (2, True, None))
        self.assertIsNone(self.lookup())

    def test_unknown_mutable_attributes_and_hooks_disable_reuse(self):
        self.store()
        self.model.external_configuration = {"scale": 1.}
        self.assertIsNone(self.lookup())
        self.store()
        self.assertIsNone(self.lookup())
        del self.model.external_configuration
        self.store()
        hook = self.model.register_forward_hook(lambda module, inputs, output: output)
        self.assertIsNone(self.lookup())
        self.store()
        self.assertIsNone(self.lookup())
        hook.remove()
        self.assertIsNone(self.lookup())

    def test_actual_resnet_configuration_and_default_cross_entropy_allow_reuse(self):
        from torchvision.models import resnet18

        self.context.model = resnet18(weights=None).eval()
        self.context.loss_function = torch.nn.CrossEntropyLoss()
        # This tests key construction only; numerical validity of a triplet is
        # the caller's responsibility and does not require a forward here.
        self.store()
        self.assertIsNotNone(self.lookup())

    def test_replacing_model_invalidates(self):
        self.store()
        self.context.model = CacheModel().eval()
        self.assertIsNone(self.lookup())

    def test_any_training_module_disables_and_clears_cache(self):
        self.store()
        self.model.child.train()
        self.assertIsNone(self.lookup())
        self.store()
        self.model.eval()
        self.assertIsNone(self.lookup())

    def test_retention_is_bounded_to_one_entry_per_mode(self):
        self.store("max")
        for seed in range(5):
            self.store(settings={**self.settings, "seed": seed})
        self.assertEqual(len(self.context._verified_singular_triplets), 2)
        self.assertIsNotNone(self.lookup("max"))
        self.assertIsNotNone(self.lookup(settings={**self.settings, "seed": 4}))
        self.assertIsNone(self.lookup(settings={**self.settings, "seed": 0}))


if __name__ == "__main__":
    unittest.main()
