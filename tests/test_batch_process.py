"""Exercise the real CLI, numerical failures, and attacks in fresh processes."""

from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from validation.generator import generate_run, save_yaml
from validation.registries.registry_builder import build_registries


REPOSITORY = Path(__file__).resolve().parents[1]

# Replace only external data and the persistent database. The script entry point,
# transformations, Jacobian products, solvers, observer, and attack are real.
CLI_WRAPPER = r"""
from pathlib import Path
import runpy
import sys
import torch
from torch.utils.data import TensorDataset
from data.dataset_def import dataset_def
from data.dataset_registry import dataset_dict
from database import connection

torch.set_num_threads(1)
schedule, database, error_log, entry_mode, *options = sys.argv[1:]
connection.DB_PATH = Path(database)
generator = torch.Generator().manual_seed(313)
images = torch.rand(8, 3, 32, 32, generator=generator)
labels = torch.arange(8)
dataset_dict['cifar10'] = dataset_def(
    'cifar10', lambda: TensorDataset(images, labels), 10, 3, (32, 32)
)
sys.argv = [
    'run.py', '--schedule', schedule, '--error-log', error_log, *options
]
if entry_mode == 'module':
    runpy.run_module('run', run_name='__main__')
else:
    runpy.run_path('run.py', run_name='__main__')
"""


class BatchProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = build_registries()

    def description(self, name, defense):
        registry = self.registry
        description = generate_run(
            registry['attacks']['dlg'], registry['defenses'][defense],
            registry['models']['lenet5'], registry['datasets']['cifar10'], .02,
        )
        description.name = name
        description.attack.optimizer_description.parameters['max_iter'] = 1
        description.observer.fail_policies[0].parameters['limit'] = 2
        return description

    def schedule(self, root):
        schedule = root / 'schedule'
        schedule.mkdir()
        ranking = self.description('real-ranking-error', 'max_svd_noise')
        ranking.gradient_transformations[0].ranking_policy.parameters['iterations'] = 1
        reference = self.description('real-reference-error', 'gaussian_noise_reference')
        reference.gradient_transformations[0].transform.parameters['scaling']['parameters'].update(
            reference_iterations=1,
        )
        healthy = self.description('real-following-attack', 'none')
        for index, description in enumerate((ranking, reference, healthy)):
            save_yaml(description, schedule).rename(schedule / f'{index:02d}.yaml')
        return schedule

    def invoke(self, root, schedule, *options, module=False):
        return subprocess.run(
            [sys.executable, '-c', CLI_WRAPPER, str(schedule), str(root/'runs.db'),
             str(root/'errors.jsonl'), 'module' if module else 'script', *options],
            cwd=REPOSITORY,
            env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            capture_output=True, text=True, timeout=90,
        )

    def assert_completed_schedule(self, process, root):
        self.assertIn('real-following-attack', process.stdout)
        self.assertIn('[CONTINUE]', process.stdout + process.stderr)
        self.assertIn('[COMPLETE]', process.stdout + process.stderr)
        with sqlite3.connect(root/'runs.db') as database:
            rows = database.execute(
                'SELECT name, outcome, summary FROM runs ORDER BY id'
            ).fetchall()
        self.assertEqual([(name, outcome) for name, outcome, _ in rows], [
            ('real-ranking-error', 'error'),
            ('real-reference-error', 'error'),
            ('real-following-attack', 'failure'),
        ])
        successful_execution = json.loads(rows[-1][2])
        self.assertEqual(successful_execution['total_iterations'], 2)
        self.assertEqual(successful_execution['reason'], 'max_iterations')
        events = [json.loads(line) for line in (root/'errors.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertTrue(event['database_recorded'])
            self.assertEqual(event['phase'], 'gradient_transformation')
            diagnostic = event['experiment_metadata']['numerical_diagnostics']['max_svd']
            self.assertEqual(diagnostic['status'], 'unconverged')
            self.assertGreater(diagnostic['relative_residual'], diagnostic['relative_tolerance'])
            self.assertIn('RuntimeError: max-SVD direction did not converge', event['traceback'])
        # Match the reference-scaling stack that originally stopped user batches.
        self.assertIn('max_svd_reference_scale.py', events[1]['traceback'])

    def test_module_continues_after_real_ranking_and_reference_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = self.invoke(root, self.schedule(root), module=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assert_completed_schedule(process, root)
            self.assertNotIn('Traceback (most recent call last)', process.stderr)

    def test_strict_exit_waits_for_remaining_attack_and_restart_skips_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = self.schedule(root)
            process = self.invoke(root, schedule, '--strict-errors')
            self.assertEqual(process.returncode, 1, process.stdout + process.stderr)
            self.assert_completed_schedule(process, root)
            repeated = self.invoke(root, schedule)
            self.assertEqual(repeated.returncode, 0, repeated.stdout + repeated.stderr)
            self.assertIn('[SKIP] real-following-attack', repeated.stdout)
            with sqlite3.connect(root/'runs.db') as database:
                attempts = database.execute(
                    'SELECT name, attempt, outcome FROM runs ORDER BY id'
                ).fetchall()
            self.assertEqual(attempts, [
                ('real-ranking-error', 1, 'error'),
                ('real-reference-error', 1, 'error'),
                ('real-following-attack', 1, 'failure'),
                ('real-ranking-error', 2, 'error'),
                ('real-reference-error', 2, 'error'),
            ])


if __name__ == '__main__':
    unittest.main()
