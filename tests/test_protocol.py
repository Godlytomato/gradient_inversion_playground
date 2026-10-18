import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from data.dataset_def import dataset_def, sample_identity, subset_identities
from models.model_def import model_definition
from models.model_registry import model_dict
from reconciliation import reconciliation
from run_utils.grad_prod import grad_prod
from supplementary_information.implementations.auxiliary_images import auxiliary_images
from supplementary_information.implementations.ground_truth import get_ground_truth
from supplementary_information.implementations.labels import get_labels
from supplementary_information.implementations.normalization import normalization


class TinyDataset(Dataset):
    targets = [0, 1, 0, 1, 2]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 32, 32), index / 4), self.targets[index]


def definition(data=None):
    return dataset_def(
        name="synthetic",
        dataset_builder=Mock(return_value=TinyDataset() if data is None else data),
        class_count=3,
        input_channels=3,
        input_dimensions=(32, 32),
        mean=(.5, .5, .5),
        std=(.25, .25, .25),
        split="test",
    )


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        random.seed(7)
        torch.manual_seed(7)

    def test_lazy_dataset_access_and_compatible_override(self):
        data = definition()
        data.dataset_builder.assert_not_called()
        self.assertIs(data.data, data.data)
        data.dataset_builder.assert_called_once()
        replacement = TinyDataset()
        data.data = replacement
        self.assertIs(data.data, replacement)
        data.dataset_builder.assert_called_once()

    def test_registry_and_generation_are_offline(self):
        # A new interpreter proves eager module initialization never downloads.
        script = """
from unittest.mock import patch
with patch('torchvision.datasets.CIFAR10', side_effect=AssertionError('dataset loaded')):
    from validation.registries.registry_builder import build_registries
    from validation.generator import generate_run
    from validation.validator import validate_run
    registries = build_registries()
    description = generate_run(registries['attacks']['dlg'], registries['defenses']['none'], registries['models']['lenet5'], registries['datasets']['cifar10'], .02)
    validate_run(description)
    from data.dataset_registry import dataset_dict
    assert dataset_dict['cifar10']._data is None
print('offline configuration passed')
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("offline configuration passed", result.stdout)

    def test_invalid_sampling_fails_before_loading(self):
        data = definition()
        for size, classes in [(0, [0]), (-1, [0]), (1.2, [0]), (True, [0]), (1, []), (1, [-1]), (1, [3]), (1, [1.5])]:
            with self.subTest(size=size, classes=classes), self.assertRaises(ValueError):
                data.sample(size, classes)
        data.dataset_builder.assert_not_called()

    def test_sampling_has_finite_eligibility_and_explicit_replacement(self):
        data = definition()
        # Only one record belongs to class 2, so replacement is observable.
        samples = data.sample(20, [2])
        self.assertEqual(samples.indices, [4] * 20)
        with self.assertRaisesRegex(ValueError, "without replacement"):
            data.sample(2, [2], replace=False)
        data.data = Subset(data.data, [0, 2])
        with self.assertRaisesRegex(ValueError, "No eligible examples"):
            data.sample(1, [1])
        excluded = {sample_identity(data.data, 0), sample_identity(data.data, 1)}
        with self.assertRaisesRegex(ValueError, "No eligible examples"):
            data.sample(1, [0], exclude_identities=excluded)

    def test_nested_subsets_exclude_private_base_records(self):
        data = definition()
        base = data.data
        # Outer index 1 -> inner index 2 -> original record 4.
        data.data = Subset(base, [3, 1, 4, 0, 2])
        private = Subset(Subset(data.data, [1, 2]), [1])
        self.assertEqual(subset_identities(private), [(id(base), 4)])
        context = SimpleNamespace(batch=private, model=nn.Linear(1, 1), experiment_metadata={"seed": 7})
        public = auxiliary_images(context, data, sample_size=100)
        self.assertFalse(bool((public == base[4][0]).flatten(1).all(dim=1).any()))
        provenance = context.experiment_metadata["auxiliary"]
        self.assertNotIn(4, provenance["base_indices"])
        self.assertEqual(provenance["excluded_private_base_indices"], [4])
        self.assertEqual(context.experiment_metadata["seed"], 7)
        json.dumps(context.experiment_metadata)

    def test_auxiliary_sampling_fails_if_private_population_exhausts_source(self):
        data = definition()
        context = SimpleNamespace(batch=Subset(data.data, range(5)), model=nn.Linear(1, 1))
        with self.assertRaisesRegex(ValueError, "No eligible examples"):
            auxiliary_images(context, data, sample_size=1)

    def test_negative_subset_indices_refer_to_the_same_private_record(self):
        data = definition()
        base = data.data
        private = Subset(Subset(base, [0, -1]), [-1])
        self.assertEqual(subset_identities(private), [(id(base), 4)])
        context = SimpleNamespace(batch=private, model=nn.Linear(1, 1))
        auxiliary_images(context, data, sample_size=30)
        self.assertNotIn(4, context.experiment_metadata["auxiliary"]["base_indices"])

    def test_unidentifiable_batches_require_a_declared_disjoint_source(self):
        data = definition()
        context = SimpleNamespace(batch=[data.data[0]], model=nn.Linear(1, 1))
        with self.assertRaisesRegex(ValueError, "Subset identities"):
            auxiliary_images(context, data)
        with self.assertRaisesRegex(ValueError, "private dataset definition"):
            auxiliary_images(context, data, public_data_def=data, public_source_disjoint=True)
        public_source = definition()
        auxiliary_images(context, data, public_data_def=public_source, public_source_disjoint=True)
        self.assertTrue(context.experiment_metadata["auxiliary"]["explicit_disjoint_source"])

    def test_image_information_matches_model_dtype_and_device(self):
        data = definition()
        context = SimpleNamespace(batch=Subset(data.data, [0]), model=nn.Linear(1, 1).double())
        self.assertEqual(get_labels(context).dtype, torch.long)
        self.assertEqual(get_ground_truth(context).dtype, torch.float64)
        self.assertEqual(auxiliary_images(context, data).dtype, torch.float64)
        self.assertEqual(normalization(context, data)["input_space"], "raw_pixels")

    def test_actual_models_match_victim_cross_entropy_gradients(self):
        data = definition()
        data.class_count = 10
        for name, model_def in model_dict.items():
            with self.subTest(model=name):
                model = reconciliation(model_def, data, model_def.model_builder()).eval()
                batch = [(torch.rand(3, 32, 32), 3), (torch.rand(3, 32, 32), 7)]
                actual = grad_prod(model, batch)
                images = torch.stack([image for image, _ in batch])
                labels = torch.tensor([label for _, label in batch])
                prediction = model(images)
                expected = torch.autograd.grad(F.cross_entropy(prediction, labels), tuple(model.parameters()))
                self.assertEqual(tuple(prediction.shape), (2, 10))
                for observed, reference in zip(actual, expected):
                    self.assertTrue(bool(observed.isfinite().all()))
                    torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    def test_fixed_shape_models_reject_unsupported_dimensions(self):
        fixed = model_definition(
            "fixed", 3, (16, 16), 3, lambda: nn.Identity(), None, None, None
        )
        with self.assertRaisesRegex(ValueError, "cannot adapt input dimensions"):
            reconciliation(fixed, definition(), fixed.model_builder())


if __name__ == "__main__":
    unittest.main()
