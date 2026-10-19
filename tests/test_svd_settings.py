"""Versioned minimum-SVD settings, historical compatibility, and schedule migration."""
import copy
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from database import connection as connection_module
from database.database import create_run, finish_run, run_exists
from database.hashing import defense_hash, run_hash
from database.schema import SCHEMA
from run_utils.reader import read_run
from run_utils.run_builder import build_run
from run_utils.svd_settings import (
    DEFAULT_MIN_SVD_METHOD, KRYLOV_DEFAULTS, LEGACY_MIN_SVD_METHOD,
    min_svd_parameters, normalize_min_svd_settings,
)
from scripts.set_min_svd_solver import configure_run
from validation.generator import generate_run
from validation.registries.registry_builder import build_registries


MIN_PRESETS = ("min_svd_noise", "min_svd_lower_bound", "min_svd_noise_reference", "min_svd_pruning")


class SvdSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registries = build_registries()

    def description(self, defense="min_svd_noise"):
        r = self.registries
        return generate_run(r["attacks"]["dlg"], r["defenses"][defense],
                            r["models"]["lenet5"], r["datasets"]["cifar10"], .02)

    @staticmethod
    def packages(description):
        transformation = description.gradient_transformations[0]
        packages = [transformation.ranking_policy.parameters]
        if transformation.transform.name == "noise":
            packages.append(transformation.transform.parameters["construction"]["parameters"])
        return packages

    def historical(self):
        description = self.description()
        for parameters in self.packages(description):
            parameters.clear()
            parameters.update(iterations=5, solve_iterations=8, damping=1e-4)
        return description

    def test_presets_choose_checked_solver_and_preserve_explicit_legacy_budgets(self):
        for name in MIN_PRESETS:
            with self.subTest(preset=name):
                preset = self.registries["defenses"][name]
                current = preset.build()[0]
                expected = {"solver_method": DEFAULT_MIN_SVD_METHOD, "solver_seed": 0,
                            **KRYLOV_DEFAULTS}
                self.assertEqual(current.ranking_policy.parameters, expected)
                if current.transform.name == "noise":
                    self.assertEqual(current.transform.parameters["construction"]["parameters"], expected)
                old = preset.build(iterations=13, solve_iterations=17, damping=.002)[0]
                self.assertEqual(old.ranking_policy.parameters, {
                    "solver_method": LEGACY_MIN_SVD_METHOD, "iterations": 13,
                    "solve_iterations": 17, "damping": .002,
                })
                custom = preset.build(krylov_max_iterations=31, krylov_check_every=7,
                                      krylov_max_elements=6000, solver_seed=11)[0]
                self.assertEqual(custom.ranking_policy.parameters["solver_seed"], 11)
                self.assertEqual(custom.ranking_policy.parameters["krylov_max_iterations"], 31)

    def test_rejects_ignored_conflicting_and_invalid_solver_settings(self):
        for parameters in (
            {"solver_method": "unknown"}, {"solver_method": ""},
            {"iterations": 0}, {"solve_iterations": True}, {"damping": float("nan")},
            {"krylov_max_iterations": 0}, {"krylov_check_every": 1.5},
            {"krylov_max_elements": True}, {"solver_seed": -1},
            {"solver_method": DEFAULT_MIN_SVD_METHOD, "iterations": 5},
            {"solver_method": LEGACY_MIN_SVD_METHOD, "solver_seed": 0},
            {"iterations": 5, "krylov_max_iterations": 64},
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                min_svd_parameters(**parameters)
        # Partial legacy settings retain the other legacy defaults.
        self.assertEqual(min_svd_parameters(iterations=7), {
            "solver_method": LEGACY_MIN_SVD_METHOD, "iterations": 7,
            "solve_iterations": 8, "damping": 1e-4,
        })

    def test_normalization_versions_old_method_without_switching_algorithm(self):
        old = self.historical()
        identity = (run_hash(old), defense_hash(old))
        self.assertIs(normalize_min_svd_settings(old), old)
        for parameters in self.packages(old):
            self.assertEqual(parameters["solver_method"], LEGACY_MIN_SVD_METHOD)
            self.assertEqual(parameters["iterations"], 5)
            self.assertNotIn("solver_seed", parameters)
        self.assertNotEqual((run_hash(old), defense_hash(old)), identity)
        once = copy.deepcopy(old)
        normalize_min_svd_settings(old)
        self.assertEqual(old, once)
        for name in ("none", "max_svd_noise", "graddefense"):
            unrelated = self.description(name)
            before = (run_hash(unrelated), defense_hash(unrelated))
            normalize_min_svd_settings(unrelated)
            self.assertEqual((run_hash(unrelated), defense_hash(unrelated)), before)

    def test_reader_and_builder_resolve_before_hashing_without_rewriting_yaml(self):
        historical = self.historical()
        old_identity = run_hash(historical)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.yaml"
            rendered = yaml.safe_dump(asdict(historical))
            path.write_text(rendered)
            loaded = read_run(str(path))
            self.assertEqual(self.packages(loaded)[0]["solver_method"], LEGACY_MIN_SVD_METHOD)
            self.assertNotEqual(run_hash(loaded), old_identity)
            self.assertEqual(path.read_text(), rendered)
        build_run(historical)
        self.assertEqual(self.packages(historical)[0]["solver_method"], LEGACY_MIN_SVD_METHOD)
        # An explicitly selected new algorithm resolves all its defaults too.
        current = self.description()
        for parameters in self.packages(current):
            parameters.clear()
            parameters["solver_method"] = DEFAULT_MIN_SVD_METHOD
        build_run(current)
        self.assertEqual(self.packages(current)[0], min_svd_parameters())

    def test_new_method_does_not_skip_or_modify_historical_results(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(connection_module, "DB_PATH", Path(folder) / "runs.db"):
                with connection_module.get_connection() as connection:
                    connection.executescript(SCHEMA)
                old = self.historical()
                old_identity = run_hash(old)
                old_id = create_run(old)
                finish_run(old_id, "failure", summary={"reason": "historical completed fixture"})
                self.assertTrue(run_exists(old_identity))
                updated = self.description()
                build_run(updated)
                self.assertFalse(run_exists(run_hash(updated)))
                new_id = create_run(updated)
                self.assertNotEqual(old_id, new_id)
                with connection_module.get_connection() as connection:
                    row = connection.execute("SELECT config_hash, outcome, raw_description FROM runs WHERE id=?",
                                             (old_id,)).fetchone()
                self.assertEqual(row[:2], (old_identity, "failure"))
                self.assertNotIn("solver_method", row[2])

    def test_migration_only_changes_minimum_solver_parameters(self):
        raw = asdict(self.historical())
        # These scientific settings must survive migration unchanged.
        construction = raw["gradient_transformations"][0]["transform"]["parameters"]["construction"]["parameters"]
        construction.update(tolerance=1e-5, exact_max_elements=123, rank_policy="error")
        original = copy.deepcopy(raw)
        updated = configure_run(raw, krylov_max_iterations=73, solver_seed=9)
        self.assertEqual(raw, original)
        self.assertEqual(updated["attack"], raw["attack"])
        self.assertEqual(updated["observer"], raw["observer"])
        transformed = updated["gradient_transformations"][0]
        old_transform = raw["gradient_transformations"][0]
        self.assertEqual(transformed["draft_policy"], old_transform["draft_policy"])
        self.assertEqual(transformed["transform"]["parameters"]["scaling"],
                         old_transform["transform"]["parameters"]["scaling"])
        construction = transformed["transform"]["parameters"]["construction"]["parameters"]
        for name, expected in (("tolerance", 1e-5), ("exact_max_elements", 123), ("rank_policy", "error")):
            self.assertEqual(construction[name], expected)
        for name in ("iterations", "solve_iterations", "damping"):
            self.assertNotIn(name, construction)
        self.assertEqual(construction["solver_method"], DEFAULT_MIN_SVD_METHOD)
        self.assertEqual(construction["krylov_max_iterations"], 73)
        self.assertEqual(construction["solver_seed"], 9)
        self.assertEqual(configure_run(updated, krylov_max_iterations=73, solver_seed=9), updated)

    def test_migration_cli_backup_dry_run_idempotence_and_all_file_prevalidation(self):
        script = Path(__file__).resolve().parents[1] / "scripts/set_min_svd_solver.py"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.yaml"
            original = yaml.safe_dump(asdict(self.historical()), sort_keys=False)
            path.write_text(original)
            unrelated = Path(folder) / "unrelated.yml"
            unrelated_text = yaml.safe_dump(asdict(self.description("none")), sort_keys=False)
            unrelated.write_text(unrelated_text)
            command = [sys.executable, str(script), folder]
            dry = subprocess.run(command + ["--dry-run"], capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertEqual(path.read_text(), original)
            self.assertFalse(path.with_name(path.name + ".before-min-svd-solver.bak").exists())
            applied = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(applied.returncode, 0, applied.stderr)
            backup = path.with_name(path.name + ".before-min-svd-solver.bak")
            self.assertEqual(backup.read_text(), original)
            configured = path.read_text()
            self.assertEqual(unrelated.read_text(), unrelated_text)
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn("0 changed", repeated.stdout)
            self.assertEqual(path.read_text(), configured)
            invalid = Path(folder) / "invalid.yml"
            invalid.write_text("observer: []\n")
            failed = subprocess.run(command + ["--solver-seed", "3"], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(path.read_text(), configured)
            self.assertEqual(backup.read_text(), original)
            self.assertFalse(invalid.with_name(invalid.name + ".before-min-svd-solver.bak").exists())


if __name__ == "__main__":
    unittest.main()
