"""CLIP scoring, label isolation, stopping and persistence without downloads."""
import importlib
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_tensor

from data.dataset_def import dataset_def
from make_run import demo_description, main as make_demo
from observer.observer import observer
from observer.policies.implementations.target_clip_confidence import target_clip_confidence
from run_def import function_package
from run_utils.reader import read_run
from run_utils.run_builder import build_run

metric = importlib.import_module("observer.metrics.implementations.clip_confidence")


class ClipMetricTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(class_count=2, class_names=("cat", "dog"))
        self.images = torch.zeros(2, 3, 4, 4)

    def score(self, probabilities, labels=(0, 1)):
        backend = SimpleNamespace(probabilities=lambda *args: torch.tensor(probabilities, dtype=torch.float64))
        with patch.object(metric, "load_classifier", return_value=backend):
            return metric.clip_confidence(self.images, labels=labels, data_def=self.data)

    def test_confident_wrong_label_is_zero_and_batch_uses_minimum(self):
        self.assertEqual(self.score([[.01, .99], [.01, .99]]), 0)
        self.assertAlmostEqual(self.score([[.99, .01], [.15, .85]]), .85)
        self.assertAlmostEqual(self.score([[.91, .09], [.01, .99]]), .91)

    def test_policy_is_inclusive_and_missing_score_does_not_pass(self):
        for value, expected in [(None, False), (0., False), (.899, False), (.90, True), (.99, True)]:
            with self.subTest(value=value):
                active = SimpleNamespace(current_metrics={} if value is None else {"clip_confidence": value})
                self.assertEqual(target_clip_confidence(active, threshold=.90), expected)

    def test_invalid_images_and_labels_fail_before_loading_weights(self):
        cases = [
            dict(reconstruction=torch.full_like(self.images, float("nan")), labels=[0, 1]),
            dict(reconstruction=self.images, labels=[0]),
            dict(reconstruction=self.images, labels=[0., 1.]),
            dict(reconstruction=self.images, labels=[0, 2]),
            dict(reconstruction=self.images[:, :2], labels=[0, 1]),
        ]
        with patch.object(metric, "load_classifier") as load:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    metric.clip_confidence(data_def=self.data, **kwargs)
            load.assert_not_called()

    def test_checkpoint_scale_temperature_preprocessing_cache_and_rng(self):
        class TinyClip(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(3.)))
                self.text_calls = 0

            def encode_text(self, tokens):
                self.text_calls += 1
                return torch.eye(2)

            def encode_image(self, images):
                return images[:, :2].mean(dim=(2, 3))

        model = TinyClip()
        prompts = []

        def tokenize(values):
            prompts.extend(values)
            return torch.arange(len(values))

        def load(*args, **kwargs):
            torch.rand(10)
            return model, to_tensor

        fake_clip = SimpleNamespace(load=load, tokenize=tokenize)
        pixels = torch.tensor([-1., 2., 0.]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4).clone()
        before_pixels = pixels.clone()
        metric.load_classifier.cache_clear()
        try:
            with patch.dict("sys.modules", {"clip": fake_clip}):
                before_rng = torch.random.get_rng_state()
                first = metric.clip_confidence(pixels, labels=[1], data_def=self.data)
                second = metric.clip_confidence(pixels, labels=[1], data_def=self.data, temperature=2.)
                self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
            self.assertAlmostEqual(first, torch.tensor([0., 3.]).softmax(0)[1].item())
            self.assertAlmostEqual(second, torch.tensor([0., 1.5]).softmax(0)[1].item())
            self.assertEqual(prompts, ["a photo of a cat", "a photo of a dog"])
            self.assertEqual(model.text_calls, 1)
            self.assertFalse(model.training)
            self.assertFalse(model.logit_scale.requires_grad)
            self.assertTrue(torch.equal(pixels, before_pixels))
        finally:
            metric.load_classifier.cache_clear()


class ClipObserverTests(unittest.TestCase):
    def description(self):
        description = demo_description()
        description.observer.logging = None
        return description

    def test_permuted_batch_matches_labels_and_preserves_raw_saved_image(self):
        description = self.description()
        config = build_run(description).observer
        truth = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
        raw = truth.flip(0)
        data = SimpleNamespace(class_count=2, class_names=("cat", "dog"))

        def probabilities(images, names, template, temperature):
            self.assertTrue(torch.equal(images, truth))
            self.assertEqual(names, ("cat", "dog"))
            return torch.tensor([[.95, .05], [.02, .98]])

        active = observer(config, ground_truth=truth, information={"labels": torch.tensor([0, 1]), "data_def": data})
        with patch.object(metric, "load_classifier", return_value=SimpleNamespace(probabilities=probabilities)):
            active.update(raw, {"attack_loss": 1.})
        self.assertEqual(active.outcome, "success")
        self.assertEqual(active.best_assignment, [1, 0])
        self.assertTrue(torch.equal(active.best_image, raw))
        self.assertTrue(active.summary()["evaluation"]["practical_success"])

    def test_final_outcome_uses_clip_and_saves_successful_candidate(self):
        config = build_run(self.description()).observer
        config.metrics = []  # Supply deterministic evaluator values to exercise selection and summary.
        active = observer(config, ground_truth=torch.zeros(1, 3, 4, 4))
        active.update(torch.zeros_like(active.ground_truth), {"clip_confidence": .7, "mse": 0., "attack_loss": .01})
        self.assertFalse(active.should_stop())
        recovered = torch.ones_like(active.ground_truth)
        active.update(recovered, {"clip_confidence": .95, "mse": 1., "attack_loss": 10.})
        self.assertEqual(active.outcome_reason, "target_clip_confidence")
        self.assertEqual(active.summary()["outcome"], "success")
        self.assertTrue(torch.equal(active.best_image, recovered))
        active.outcome = "error"
        self.assertEqual(active.summary()["outcome"], "error")

    def test_below_threshold_fails_at_cap_even_with_perfect_mse(self):
        config = build_run(self.description()).observer
        config.metrics = []
        config.fail_policies[0].parameters["limit"] = 1
        active = observer(config, ground_truth=torch.zeros(1, 3, 4, 4))
        active.update(active.ground_truth.clone(), {"clip_confidence": .89, "mse": 0., "attack_loss": 1.})
        self.assertEqual(active.outcome_reason, "max_iterations")
        self.assertEqual(active.summary()["outcome"], "failure")

    def test_clip_information_stays_with_evaluator_and_validation_is_offline(self):
        with patch.object(metric, "load_classifier") as load:
            executable = build_run(self.description())
        self.assertIn("labels", executable.requirements.observer_information)
        self.assertIn("data_def", executable.requirements.observer_information)
        self.assertNotIn("labels", executable.requirements.attack_information)
        self.assertNotIn("ground_truth", executable.requirements.attack_information)
        load.assert_not_called()

    def test_invalid_thresholds_temperatures_and_missing_metric_are_rejected(self):
        for threshold in [0, -1, 1.1, float("nan"), float("inf"), True]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                demo_description(threshold=threshold)
        for temperature in [0, -1, float("nan"), float("inf"), True]:
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                demo_description(temperature=temperature)
        description = self.description()
        description.observer.metrics = [function_package("mse")]
        with self.assertRaises(ValueError):
            build_run(description)
        description = self.description()
        description.observer.evaluation_target_mse = .02
        with self.assertRaises(ValueError):
            build_run(description)

    def test_single_demo_round_trips_and_rejects_unrelated_schedule_files(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(make_demo(["--output", folder, "--iterations", "2"]), 0)
            files = list(Path(folder).glob("*.yaml"))
            self.assertEqual(len(files), 1)
            description = read_run(str(files[0]))
            self.assertEqual(description.observer.evaluation_target_clip_confidence, .9)
            self.assertEqual(description.observer.fail_policies[0].parameters["limit"], 2)
            build_run(description)
            (Path(folder) / "other.yaml").write_text("{}")
            with self.assertRaises(SystemExit):
                make_demo(["--output", folder])

    def test_unused_clip_target_preserves_existing_configuration_hashes(self):
        from database.hashing import config_hash

        description = self.description()
        description.observer.evaluation_target_clip_confidence = None
        previous = asdict(description)
        previous["observer"].pop("evaluation_target_clip_confidence")
        self.assertEqual(config_hash(description), config_hash(previous))
        description.observer.evaluation_target_clip_confidence = .9
        self.assertNotEqual(config_hash(description), config_hash(previous))

    def test_real_attack_runner_logs_metric_and_clip_outcome(self):
        import run as runner
        from database import connection
        from database.setup import setup_database
        from data.dataset_registry import dataset_dict

        generator = torch.Generator().manual_seed(9)
        images = torch.rand(4, 3, 32, 32, generator=generator)
        fixture = dataset_def("cifar10", lambda: TensorDataset(images, torch.zeros(4, dtype=torch.long)),
                              10, 3, (32, 32), class_names=tuple(str(i) for i in range(10)))
        probabilities = torch.full((1, 10), .05 / 9)
        probabilities[0, 0] = .95
        backend = SimpleNamespace(probabilities=lambda *args: probabilities)
        description = demo_description(iterations=2)
        with tempfile.TemporaryDirectory() as folder, patch.object(connection, "DB_PATH", Path(folder) / "test.db"), \
                patch.dict(dataset_dict, {"cifar10": fixture}), \
                patch.object(metric, "load_classifier", return_value=backend):
            setup_database()
            self.assertIsNotNone(runner.execute_description(description, skip_existing=False))
            with sqlite3.connect(connection.DB_PATH) as db:
                outcome, encoded = db.execute("SELECT outcome, summary FROM runs").fetchone()
                summary = json.loads(encoded)
                self.assertEqual(outcome, "success")
                self.assertEqual(summary["reason"], "target_clip_confidence")
                self.assertEqual(summary["total_iterations"], 1)
                self.assertGreaterEqual(summary["best_metrics"]["clip_confidence"], .9)
                self.assertEqual(summary["evaluation"]["success_metric"], "clip_confidence")
                self.assertIsNone(summary["evaluation"]["target_mse"])
                recorded = json.loads(db.execute("SELECT metrics FROM logs").fetchone()[0])
                self.assertGreaterEqual(recorded["clip_confidence"], .9)


if __name__ == "__main__":
    unittest.main()
