import copy
from dataclasses import asdict
from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import json
import sqlite3

import torch
from torch.utils.data import TensorDataset

from data.dataset_def import dataset_def
from data.dataset_registry import dataset_dict
from database.hashing import run_hash
from run_def import function_package, logging_description
from run_utils.reader import read_run
from run_utils.run_builder import build_run
from validation.generator import generate_run, save_yaml
from validation.registries.registry_builder import build_registries
import run as runner


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.registries = build_registries()

    def description(self, defense="none"):
        r = self.registries
        return generate_run(r["attacks"]["dlg"], r["defenses"][defense],
                            r["models"]["lenet5"], r["datasets"]["cifar10"], .02)

    def test_generated_protocol_is_fixed_budget_with_separate_oracle(self):
        for attack in self.registries["attacks"].values():
            d = self.description()
            d = generate_run(attack, self.registries["defenses"]["none"],
                             self.registries["models"]["lenet5"],
                             self.registries["datasets"]["cifar10"], .03)
            self.assertEqual(d.observer.success_policies, [])
            self.assertEqual([(p.name, p.parameters) for p in d.observer.fail_policies],
                             [("max_iterations", {"limit": attack.max_iterations})])
            self.assertEqual(d.observer.best_selection.metric, "attack_loss")
            self.assertEqual(d.observer.oracle_selection.metric, "mse")
            self.assertEqual(d.observer.evaluation_target_mse, .03)
            self.assertEqual(d.observer.batch_matching, "assignment")
            build_run(d)

    def test_every_description_round_trips_and_builds_offline(self):
        r = self.registries
        count = 0
        with tempfile.TemporaryDirectory() as tmp:
            for a, d, m, ds in product(*(r[k].values() for k in ("attacks", "defenses", "models", "datasets"))):
                description = generate_run(a, d, m, ds, .02)
                build_run(description)
                path = save_yaml(description, Path(tmp))
                self.assertEqual(run_hash(description), run_hash(read_run(str(path))))
                count += 1
        self.assertEqual(count, 9 * 26 * 3)

    def test_generated_target_reaches_calibrated_defenses(self):
        import inspect
        r = self.registries
        for defense in r["defenses"].values():
            if "target_error" not in inspect.signature(defense.builder).parameters:
                continue
            with self.subTest(defense=defense.name):
                d = generate_run(r["attacks"]["dlg"], defense, r["models"]["lenet5"],
                                 r["datasets"]["cifar10"], .1)
                self.assertEqual(d.observer.evaluation_target_mse, .1)
                scalers = [item.transform.parameters["scaling"] for item in d.gradient_transformations
                           if item.transform.name == "noise"]
                self.assertTrue(scalers)
                for scaler in scalers:
                    self.assertEqual(scaler["parameters"]["target_error"], .1)

    def test_invalid_stopping_logging_transforms_and_labels_rejected(self):
        cases = []
        d = self.description(); d.attack.restarts = 0; cases.append(d)
        d = self.description(); d.observer = None; cases.append(d)
        d = self.description(); d.observer.fail_policies = []; cases.append(d)
        d = self.description(); d.observer.logging.metrics.append("typo"); cases.append(d)
        d = self.description(); d.observer.metrics = [function_package("mse")]
        d.observer.success_policies = [function_package("target_psnr", {"threshold": 30})]; cases.append(d)
        d = self.description(); d.observer.fail_policies.append(function_package("no_improvement", {"metric": "typo"})); cases.append(d)
        d = self.description("random_pruning"); d.gradient_transformations[0].transform.name = "typo"; cases.append(d)
        d = self.description("soteria"); d.gradient_transformations[0].draft_policy.name = "highest_draft"; cases.append(d)
        for policy in ("idlg", "hf_gradinv"):
            d = self.description(); d.attack.label_policy = policy; d.data.size = 2; cases.append(d)
        for index, d in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_run(d)

    def test_real_runner_is_reproducible_and_pairs_victim_state(self):
        generator = torch.Generator().manual_seed(987)
        images = torch.rand(24, 3, 32, 32, generator=generator)
        labels = torch.arange(24) % 10
        fixture = dataset_def("cifar10", lambda: TensorDataset(images, labels), 10, 3, (32, 32))
        captured = []
        real_handle = runner.handle_run

        def capture(executable, context):
            result = real_handle(executable, context)
            captured.append(copy.deepcopy(context.experiment_metadata))
            return result

        def small_description(defense="none"):
            d = self.description(defense)
            d.experiment_seed = 23
            d.observer.logging = logging_description("none", [])
            d.observer.fail_policies[0].parameters["limit"] = 2
            d.attack.optimizer_description.parameters["max_iter"] = 1
            return d

        with patch.dict(dataset_dict, {"cifar10": fixture}), patch.object(runner, "handle_run", capture):
            first = runner.execute_description(small_description(), skip_existing=False)
            random.seed(555); torch.manual_seed(555)
            second = runner.execute_description(small_description(), skip_existing=False)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(captured[0], captured[1])
            runner.execute_description(small_description("gaussian_noise"), skip_existing=False)
            self.assertEqual(captured[0]["model_state_sha256"], captured[2]["model_state_sha256"])
            self.assertEqual(captured[0]["sample_indices"], captured[2]["sample_indices"])
            changed = small_description(); changed.experiment_seed += 1
            runner.execute_description(changed, skip_existing=False)
            self.assertNotEqual(captured[0]["model_state_sha256"], captured[3]["model_state_sha256"])

    def test_runner_persists_both_selections_and_protocol_in_temporary_database(self):
        from database import connection
        from database.setup import setup_database
        images = torch.linspace(0, 1, 12 * 3 * 32 * 32).reshape(12, 3, 32, 32)
        fixture = dataset_def("cifar10", lambda: TensorDataset(images, torch.arange(12) % 10), 10, 3, (32, 32))
        d = self.description()
        d.observer.fail_policies[0].parameters["limit"] = 2
        d.attack.optimizer_description.parameters["max_iter"] = 1
        with tempfile.TemporaryDirectory() as tmp, patch.object(connection, "DB_PATH", Path(tmp) / "test.db"), patch.dict(dataset_dict, {"cifar10": fixture}):
            setup_database()
            self.assertIsNotNone(runner.execute_description(d))
            self.assertIsNone(runner.execute_description(d))
            with sqlite3.connect(connection.DB_PATH) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
                outcome, summary = db.execute("SELECT outcome, summary FROM runs").fetchone()
                result = json.loads(summary)
                self.assertIn(outcome, {"success", "failure"})
                self.assertIn("attack_loss", result["best_metrics"])
                self.assertIn("mse", result["oracle_best_metrics"])
                self.assertEqual(result["total_iterations"], 2)
                self.assertEqual(result["reason"], "max_iterations")
                self.assertEqual(result["experiment_metadata"]["protocol_version"], 2)
                kinds = {row[0] for row in db.execute("SELECT image_type FROM images")}
                self.assertTrue({"ground_truth", "best_reconstruction", "oracle_reconstruction"}.issubset(kinds))


if __name__ == "__main__":
    unittest.main()
