"""Configuration identities and YAML compatibility for revised bound semantics."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from database import connection as connection_module
from database.database import create_run, finish_run, run_exists
from database.hashing import defense_hash, run_hash
from database.schema import SCHEMA
from run_utils.bound_settings import (
    DEFAULT_BOUND_METHOD, LEGACY_BOUND_METHOD, normalize_bound_settings,
)
from run_utils.reader import read_run
from run_utils.run_builder import build_run
from validation.generator import generate_run
from validation.registries.registry_builder import build_registries


BOUND_PRESETS = (
    "max_svd_lower_bound", "min_svd_lower_bound", "min_svd_noise_reference",
    "gaussian_noise_reference", "laplace_noise_reference",
    "uniform_noise_reference", "graddefense_reference",
)


class BoundSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registries = build_registries()

    def description(self, defense="max_svd_lower_bound"):
        r = self.registries
        return generate_run(r["attacks"]["dlg"], r["defenses"][defense],
                            r["models"]["lenet5"], r["datasets"]["cifar10"], .02)

    @staticmethod
    def settings(description):
        return description.gradient_transformations[0].transform.parameters["scaling"]["parameters"]

    def historical_description(self):
        description = self.description()
        settings = self.settings(description)
        for name in ("bound_method", "residual_l2", "residual_fraction"):
            settings.pop(name, None)
        return description

    def test_all_bound_presets_serialize_conditional_residual_assumptions(self):
        for name in BOUND_PRESETS:
            with self.subTest(preset=name):
                preset = self.registries["defenses"][name]
                transformation = preset.build(mu_j=0.)[0]
                settings = transformation.transform.parameters["scaling"]["parameters"]
                self.assertEqual(settings["bound_method"], DEFAULT_BOUND_METHOD)
                self.assertEqual(settings["residual_l2"], 0.)
                self.assertEqual(settings["residual_fraction"], .1)
                self.assertEqual(settings["mu_j"], 0.)
                explicit = preset.build(residual_l2=.2, residual_fraction=.3)[0]
                params = explicit.transform.parameters["scaling"]["parameters"]
                self.assertEqual(params["residual_l2"], .2)
                self.assertEqual(params["residual_fraction"], .3)
                legacy = preset.build(bound_method=LEGACY_BOUND_METHOD)[0]
                params = legacy.transform.parameters["scaling"]["parameters"]
                self.assertEqual(params["bound_method"], LEGACY_BOUND_METHOD)
                self.assertNotIn("residual_l2", params)
                self.assertNotIn("residual_fraction", params)

    def test_presets_reject_nonfinite_constants_and_residual_conditions(self):
        for name in BOUND_PRESETS:
            for parameter, value in (("mu_j", float("nan")), ("mu_j", -1.),
                                     ("mu_j", float("inf")),
                                     ("residual_l2", -.1),
                                     ("residual_fraction", float("inf")),
                                     ("residual_fraction", 1.),
                                     ("bound_method", "unknown")):
                with self.subTest(preset=name, parameter=parameter, value=value):
                    with self.assertRaises(ValueError):
                        self.registries["defenses"][name].build(**{parameter: value})

    def test_normalization_changes_only_affected_identities_and_is_idempotent(self):
        original = self.historical_description()
        old_run_hash, old_defense_hash = run_hash(original), defense_hash(original)
        self.assertIs(normalize_bound_settings(original), original)
        self.assertNotEqual(run_hash(original), old_run_hash)
        self.assertNotEqual(defense_hash(original), old_defense_hash)
        once = copy.deepcopy(original)
        normalize_bound_settings(original)
        self.assertEqual(original, once)
        for defense in ("none", "max_svd_noise", "graddefense"):
            description = self.description(defense)
            before = (run_hash(description), defense_hash(description))
            normalize_bound_settings(description)
            self.assertEqual((run_hash(description), defense_hash(description)), before)

    def test_reader_resolves_existing_yaml_before_standalone_hashing(self):
        original = self.historical_description()
        historical_hash = run_hash(original)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yaml"
            path.write_text(yaml.safe_dump(asdict(original)))
            loaded = read_run(str(path))
            self.assertNotEqual(run_hash(loaded), historical_hash)
            self.assertEqual(self.settings(loaded)["bound_method"], DEFAULT_BOUND_METHOD)
            # The migration is in memory; the user's schedule is preserved.
            self.assertNotIn("bound_method", path.read_text())

    def test_build_resolves_direct_descriptions_before_skip_identity(self):
        description = self.historical_description()
        original_hash = run_hash(description)
        build_run(description)
        self.assertNotEqual(run_hash(description), original_hash)
        self.assertEqual(self.settings(description)["residual_fraction"], .1)

    def test_explicit_legacy_and_corrected_settings_are_preserved(self):
        for method in (LEGACY_BOUND_METHOD, DEFAULT_BOUND_METHOD):
            description = self.historical_description()
            params = self.settings(description)
            params["bound_method"] = method
            if method == DEFAULT_BOUND_METHOD:
                params.update(residual_l2=.25, residual_fraction=.3)
            before = copy.deepcopy(params)
            normalize_bound_settings(description)
            self.assertEqual(self.settings(description), before)

    def test_completed_historical_bound_does_not_skip_corrected_or_modify_old_row(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "runs.db"
            with patch.object(connection_module, "DB_PATH", database):
                with connection_module.get_connection() as connection:
                    connection.executescript(SCHEMA)
                old = self.historical_description()
                old_identity = run_hash(old)
                old_id = create_run(old)
                finish_run(old_id, "failure", summary={"reason": "historical fixture"})
                self.assertTrue(run_exists(old_identity))
                corrected = copy.deepcopy(old)
                normalize_bound_settings(corrected)
                self.assertFalse(run_exists(run_hash(corrected)))
                new_id = create_run(corrected)
                self.assertNotEqual(new_id, old_id)
                with connection_module.get_connection() as connection:
                    row = connection.execute(
                        "SELECT config_hash, outcome, raw_description FROM runs WHERE id=?",
                        (old_id,),
                    ).fetchone()
                self.assertEqual(row[:2], (old_identity, "failure"))
                self.assertNotIn("bound_method", row[2])

    def test_invalid_forwarded_reference_parameters_fail_before_execution(self):
        for change in ({'residul_fraction': .1}, {'residual_fraction': 1.}):
            description = self.description('gaussian_noise_reference')
            self.settings(description).update(change)
            with self.assertRaises(ValueError):
                build_run(description)


if __name__ == "__main__":
    unittest.main()
