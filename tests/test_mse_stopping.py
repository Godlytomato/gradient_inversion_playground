import copy
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from dacite import from_dict
import torch
import yaml

from observer.observer import observer
from run_def import run_desc
from run_utils.run_builder import build_run
from scripts.set_mse_stopping import configure_run
from validation.generator import generate_run
from validation.registries.registry_builder import build_registries


class MseStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registries = build_registries()

    def raw(self):
        r = self.registries
        return asdict(generate_run(r['attacks']['dlg'], r['defenses']['none'],
                                   r['models']['lenet5'], r['datasets']['cifar10'], .02))

    def active(self, patience=3):
        raw = configure_run(self.raw(), patience=patience)
        raw['observer']['logging'] = None
        config = build_run(from_dict(data_class=run_desc, data=raw)).observer
        return observer(config, ground_truth=torch.zeros(1, 1, 2, 2))

    def test_stops_and_saves_successful_best_mse_image(self):
        active = self.active()
        active.update(torch.full((1, 1, 2, 2), .5), {'attack_loss': .01})
        self.assertFalse(active.should_stop())
        recovered = torch.full((1, 1, 2, 2), .1)
        active.update(recovered, {'attack_loss': 10.})
        self.assertTrue(active.should_stop())
        self.assertEqual(active.outcome_reason, 'target_mse')
        self.assertEqual(active.summary()['outcome'], 'success')
        torch.testing.assert_close(active.best_image, recovered)
        policy = active.config.success_policies[0]
        self.assertTrue(policy.function(SimpleNamespace(current_metrics={'mse': .02}), **policy.parameters))

    def test_stalls_fail_after_patience_and_retains_iteration_cap(self):
        active = self.active(patience=3)
        for _ in range(3):
            active.update(torch.full((1, 1, 2, 2), .5), {'attack_loss': 1.})
            self.assertFalse(active.should_stop())
        active.update(torch.full((1, 1, 2, 2), .5), {'attack_loss': 1.})
        self.assertTrue(active.should_stop())
        self.assertEqual(active.outcome_reason, 'no_improvement')
        self.assertEqual(active.summary()['outcome'], 'failure')
        raw = self.raw()
        updated = configure_run(raw)
        self.assertEqual(updated['observer']['fail_policies'][-1], raw['observer']['fail_policies'][0])
        self.assertEqual(updated['attack'], raw['attack'])
        self.assertEqual(updated['gradient_transformations'], raw['gradient_transformations'])

    def test_cli_backup_dry_run_idempotence_and_preflight_validation(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/set_mse_stopping.py'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.yaml'
            original = yaml.safe_dump(self.raw(), sort_keys=False)
            path.write_text(original)
            command = [sys.executable, str(script), folder, '--patience', '3']
            dry = subprocess.run(command + ['--dry-run'], text=True, capture_output=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertEqual(path.read_text(), original)
            applied = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(applied.returncode, 0, applied.stderr)
            backup = path.with_name(path.name + '.before-mse-stop.bak')
            self.assertEqual(backup.read_text(), original)
            configured = path.read_text()
            repeated = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertIn('0 changed', repeated.stdout)
            self.assertEqual(path.read_text(), configured)
            # An invalid sibling must prevent updates to otherwise valid files.
            invalid = Path(folder) / 'invalid.yml'
            invalid.write_text('observer: []\n')
            failed = subprocess.run(command + ['--target', '.03'], text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(path.read_text(), configured)
            self.assertEqual(backup.read_text(), original)

    def test_rejects_invalid_settings_without_mutating_input(self):
        raw = self.raw()
        before = copy.deepcopy(raw)
        for parameters in [{'target': float('nan')}, {'patience': 0}, {'minimum_change': 0}]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                configure_run(raw, **parameters)
        self.assertEqual(raw, before)


if __name__ == '__main__':
    unittest.main()
