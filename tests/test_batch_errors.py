import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset
import yaml

from data.dataset_def import dataset_def
from data.dataset_registry import dataset_dict
from database import connection
from run_def import function_package
from validation.generator import generate_run, save_yaml
from validation.registries.registry_builder import build_registries
import run as runner


class BatchErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.registry = build_registries()

    def description(self, name, defense='none'):
        r = self.registry
        d = generate_run(r['attacks']['dlg'], r['defenses'][defense],
                         r['models']['lenet5'], r['datasets']['cifar10'], .02)
        d.name = name
        d.attack.optimizer_description.parameters['max_iter'] = 1
        d.observer.fail_policies[0].parameters['limit'] = 2
        return d

    @contextlib.contextmanager
    def batch(self, descriptions):
        generator = torch.Generator().manual_seed(313)
        fixture = dataset_def('cifar10', lambda: TensorDataset(
            torch.rand(8, 3, 32, 32, generator=generator), torch.arange(8)),
            10, 3, (32, 32))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schedule = root / 'schedule'; schedule.mkdir()
            for index, d in enumerate(descriptions):
                path = save_yaml(d, schedule)
                path.rename(schedule / f'{index:02d}.yaml')
            with patch.object(connection, 'DB_PATH', root / 'runs.db'), patch.dict(dataset_dict, {'cifar10': fixture}):
                yield root, schedule

    def invoke(self, root, schedule, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return runner.main(['--schedule', str(schedule), '--error-log', str(root/'errors.jsonl'), *extra])

    def test_transform_error_keeps_diagnostics_and_continues_normal_failures(self):
        bad = self.description('numerical-error', 'max_svd_noise')
        healthy = self.description('budget-failure')
        stalled = self.description('stall-failure')
        stalled.attack.optimizer_description = function_package('adam', {'lr': 0.0})
        stalled.observer.fail_policies = [
            function_package('no_improvement', {'metric': 'mse', 'patience': 1}),
            function_package('max_iterations', {'limit': 3}),
        ]
        original = runner.transformer

        def fail_one(grads, transformations, context):
            if transformations:
                context.numerical_diagnostics['max_svd'] = {
                    'status': 'unconverged', 'relative_residual': .000795,
                }
                raise RuntimeError('max-SVD direction did not converge')
            return original(grads, transformations, context)

        with self.batch([bad, healthy, stalled]) as (root, schedule), patch.object(runner, 'transformer', fail_one):
            self.assertEqual(self.invoke(root, schedule), 1)
            with sqlite3.connect(connection.DB_PATH) as db:
                rows = {name: (outcome, json.loads(summary)) for name, outcome, summary in
                        db.execute('SELECT name,outcome,summary FROM runs')}
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows['numerical-error'][0], 'error')
                detail = rows['numerical-error'][1]
                self.assertEqual(detail['phase'], 'gradient_transformation')
                self.assertEqual(detail['experiment_metadata']['numerical_diagnostics']['max_svd']['relative_residual'], .000795)
                self.assertEqual(rows['budget-failure'][0], 'failure')
                self.assertEqual(rows['budget-failure'][1]['reason'], 'max_iterations')
                self.assertEqual(rows['stall-failure'][0], 'failure')
                self.assertEqual(rows['stall-failure'][1]['reason'], 'no_improvement')
            errors = [json.loads(line) for line in (root/'errors.jsonl').read_text().splitlines()]
            self.assertEqual(len(errors), 1)
            self.assertTrue(errors[0]['database_recorded'])
            self.assertIn('RuntimeError', errors[0]['traceback'])
            # A later launch retries the numerical error and skips normal failures.
            self.assertEqual(self.invoke(root, schedule), 1)
            with sqlite3.connect(connection.DB_PATH) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 4)

    def test_attack_error_is_recorded_once_and_next_run_executes(self):
        bad = self.description('optimizer-error')
        bad.attack.optimizer_description = function_package('adam', {'lr': -1.0})
        healthy = self.description('next-run')
        with self.batch([bad, healthy]) as (root, schedule):
            self.assertEqual(self.invoke(root, schedule), 1)
            with sqlite3.connect(connection.DB_PATH) as db:
                rows = db.execute('SELECT name,outcome FROM runs ORDER BY id').fetchall()
                self.assertEqual(rows, [('optimizer-error', 'error'), ('next-run', 'failure')])
            event = json.loads((root/'errors.jsonl').read_text())
            self.assertTrue(event['database_recorded'])

    def test_malformed_yaml_is_logged_and_next_run_executes(self):
        with self.batch([self.description('next-run')]) as (root, schedule):
            (schedule/'-invalid.yaml').write_text('observer: [\n')
            self.assertEqual(self.invoke(root, schedule), 1)
            event = json.loads((root/'errors.jsonl').read_text())
            self.assertEqual(event['phase'], 'read_description')
            self.assertFalse(event['database_recorded'])
            with sqlite3.connect(connection.DB_PATH) as db:
                self.assertEqual(db.execute('SELECT name FROM runs').fetchall(), [('next-run',)])

    def test_fail_fast_and_keyboard_interrupt_remain_available(self):
        with self.batch([self.description('next-run')]) as (root, schedule):
            invalid = schedule/'-invalid.yaml'; invalid.write_text('observer: [\n')
            with self.assertRaises(yaml.YAMLError):
                self.invoke(root, schedule, '--fail-fast')
            invalid.unlink()
            with patch.object(runner, 'execute_description', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.invoke(root, schedule)


if __name__ == '__main__':
    unittest.main()
