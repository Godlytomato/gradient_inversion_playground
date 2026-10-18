"""Regression checks for evaluation, final summaries and safe persistence."""
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from database import connection as connection_module
from database.database import create_run, finish_run, run_exists, add_log, get_run_images
from database.hashing import config_hash, run_hash
from database.image_display import viewable, align_for_display
from database.export_images import export_images
from database.viewer import render_run
from database.schema import SCHEMA
from database.setup import main as setup_main
from database.scripts.heatmap import build_matrix, summarize_results
from observer.batch_matching import align_batch, minimum_assignment
from observer.metrics.implementations.mse import mse
from observer.metrics.implementations.psnr import psnr
from observer.observer import observer
from observer.logging.implementations.database import database_logger
from observer.policies.implementations.target_mse import target_mse
from observer.policies.implementations.target_psnr import target_psnr
from observer.policies.implementations.max_iterations import max_iterations
from run_def import (resolved_observer, resolved_function, best_selection_description,
                     function_package, run_desc, batch)
from validation.implementations.attacks.dlg import dlg


def description():
    return run_desc('fixture', 'lenet5', batch('synthetic', 2, [0]), [], dlg(),
                    attack_name='dlg', defense_name='none')


def observer_config(**extra):
    values = dict(metrics=[resolved_function(mse, name='mse')], success_policies=[],
                  fail_policies=[], best_selection=best_selection_description('mse'))
    values.update(extra)
    return resolved_observer(**values)


class EvaluationTests(unittest.TestCase):
    def test_exact_permutations_are_successful_and_raw_image_is_preserved(self):
        truth = torch.arange(4.).reshape(4, 1, 1, 1)
        raw = truth[[2, 0, 3, 1]].clone()
        obs = observer(observer_config(success_policies=[resolved_function(target_mse, {'threshold': 0})]), ground_truth=truth)
        obs.update(raw)
        self.assertEqual(obs.current_metrics['mse'], 0)
        self.assertEqual(obs.outcome, 'success')
        self.assertTrue(torch.equal(obs.best_image, raw))
        aligned = align_for_display(obs.best_image, 'best_reconstruction', obs.summary())
        self.assertTrue(torch.equal(aligned, truth))

    def test_hungarian_matches_exhaustive_minimum_and_ordered_mode(self):
        generator = torch.Generator().manual_seed(5)
        for n in range(1, 6):
            cost = torch.randn(n, n, generator=generator).tolist()
            selected = minimum_assignment(cost)
            actual = sum(cost[i][selected[i]] for i in range(n))
            expected = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(actual, expected)
        truth = torch.tensor([0., 1.]).reshape(2, 1, 1, 1)
        ordered, mapping = align_batch(truth.flip(0), truth, 'ordered')
        self.assertEqual(mse(ordered, truth), 1)
        self.assertEqual(mapping, [0, 1])

    def test_perfect_psnr_is_a_valid_optimum_and_success(self):
        truth = torch.ones(1, 1, 4, 4)
        obs = observer(observer_config(metrics=[resolved_function(psnr, name='psnr')],
                       best_selection=best_selection_description('psnr', 'maximize'),
                       success_policies=[resolved_function(target_psnr, {'threshold': 30})]), ground_truth=truth)
        obs.update(truth.clone())
        self.assertEqual(obs.best_metrics['psnr'], float('inf'))
        self.assertEqual(obs.outcome, 'success')
        with self.assertRaises(ValueError):
            obs.update(torch.full_like(truth, float('nan')))

    def test_practical_and_oracle_selection_are_separate_and_truth_never_stops(self):
        obs = observer(observer_config(best_selection=best_selection_description('attack_loss'),
                       oracle_selection=best_selection_description('mse'), evaluation_target_mse=0.01,
                       fail_policies=[resolved_function(max_iterations, {'limit': 2})]),
                       ground_truth=torch.zeros(1, 1, 1, 1))
        obs.update(torch.zeros(1, 1, 1, 1), {'attack_loss': 2.0})
        self.assertFalse(obs.should_stop())
        obs.update(torch.ones(1, 1, 1, 1), {'attack_loss': 1.0})
        self.assertEqual(obs.best_metrics['mse'], 1)
        self.assertEqual(obs.oracle_best_metrics['mse'], 0)
        summary = obs.summary()
        self.assertFalse(summary['evaluation']['practical_success'])
        self.assertTrue(summary['evaluation']['oracle_success'])
        self.assertEqual(summary['outcome'], 'failure')
        self.assertEqual(summary['best_step'], 2)
        self.assertEqual(summary['oracle_best_step'], 1)

    def test_final_target_changes_outcome_only_after_fixed_budget(self):
        obs = observer(observer_config(evaluation_target_mse=0.01,
                       fail_policies=[resolved_function(max_iterations, {'limit': 2})]),
                       ground_truth=torch.zeros(1, 1, 1, 1))
        obs.update(torch.zeros(1, 1, 1, 1))
        self.assertEqual(obs.outcome, 'running')
        obs.update(torch.zeros(1, 1, 1, 1))
        self.assertEqual(obs.outcome, 'failure')
        self.assertEqual(obs.summary()['outcome'], 'success')
        self.assertEqual(obs.summary()['stopping_outcome'], 'failure')
        self.assertEqual(obs.summary()['reason'], 'max_iterations')

    def test_display_preserves_raw_colors_and_fixed_range(self):
        np.testing.assert_array_equal(viewable(torch.zeros(3, 2, 2)), np.zeros((2, 2, 3)))
        np.testing.assert_array_equal(viewable(torch.ones(3, 2, 2)), np.ones((2, 2, 3)))
        np.testing.assert_allclose(viewable(torch.tensor([[[-1., .5, 2.]]])), [[0, .5, 1]])


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'fixture.db'
        self.patches = [patch('database.connection.DB_PATH', self.path), patch('database.setup.DB_PATH', self.path)]
        for p in self.patches:
            p.start()
        with connection_module.get_connection() as connection:
            connection.executescript(SCHEMA)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.directory.cleanup()

    def test_retry_creates_preserved_attempt_and_terminal_result_deduplicates(self):
        desc = description()
        first = create_run(desc)
        add_log(first, 1, {'mse': 1})
        finish_run(first, 'error', summary={'reason': 'transient fixture'})
        self.assertFalse(run_exists(run_hash(desc)))
        second = create_run(desc)
        add_log(second, 1, {'mse': 0})
        self.assertNotEqual(first, second)
        with connection_module.get_connection() as connection:
            rows = connection.execute('SELECT id, run_hash, config_hash, attempt, summary FROM runs ORDER BY id').fetchall()
            self.assertEqual([row[3] for row in rows], [1, 2])
            self.assertEqual(rows[0][1], run_hash(desc))
            self.assertEqual(rows[1][2], run_hash(desc))
            self.assertEqual(json.loads(rows[0][4])['reason'], 'transient fixture')
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM logs').fetchone()[0], 2)
        finish_run(second, 'failure')
        self.assertTrue(run_exists(run_hash(desc)))
        with self.assertRaises(ValueError):
            create_run(desc)

    def test_incomplete_and_legacy_are_distinct(self):
        desc = description()
        original = create_run(desc)
        self.assertFalse(run_exists(run_hash(desc)))
        retry = create_run(desc)
        self.assertNotEqual(original, retry)
        # A historical success must not suppress a corrected v2 experiment.
        with connection_module.get_connection() as connection:
            connection.execute('UPDATE runs SET run_hash=?, config_hash=NULL, outcome=? WHERE id=?',
                               (run_hash(desc).removeprefix('v2:'), 'success', original))
        self.assertFalse(run_exists(run_hash(desc)))

    def test_full_summary_and_both_raw_images_round_trip(self):
        desc = description()
        logger = database_logger(['mse', 'psnr'], run_hash(desc), desc)
        logger.log(1, {'mse': 0, 'psnr': float('inf')})
        summary = dict(outcome='error', reason='RuntimeError: exact fixture', total_iterations=17,
                       best_metrics={'mse': 0, 'psnr': float('inf')}, best_assignment=[1, 0],
                       oracle_best_metrics={'mse': 0}, run_metrics={'budget': 0.1},
                       experiment_metadata={'seed': 3, 'indices': [4, 5]})
        image = torch.zeros(2, 1, 2, 2)
        logger.save_summary(desc, summary, best_image=image, ground_truth=image, oracle_image=image + .1)
        with connection_module.get_connection() as connection:
            row = connection.execute('SELECT summary, run_metrics, outcome FROM runs').fetchone()
            saved = json.loads(row[0])
            self.assertEqual(saved['reason'], summary['reason'])
            self.assertEqual(saved['total_iterations'], 17)
            self.assertEqual(saved['experiment_metadata'], summary['experiment_metadata'])
            self.assertEqual(saved['best_metrics']['psnr'], 'Infinity')
            self.assertEqual(json.loads(row[1]), summary['run_metrics'])
            self.assertEqual(row[2], 'error')
        images = get_run_images(logger.database_run_id)
        self.assertEqual(set(images), {'best_reconstruction', 'oracle_reconstruction', 'ground_truth'})
        self.assertTrue(torch.equal(images['best_reconstruction'], image))

    def test_viewer_and_export_render_preserved_raw_images(self):
        desc = description()
        logger = database_logger(['mse'], run_hash(desc), desc)
        image = torch.stack([torch.zeros(3, 2, 2), torch.ones(3, 2, 2)])
        summary = dict(outcome='success', run_metrics={}, best_assignment=[1, 0])
        logger.save_summary(desc, summary, best_image=image.flip(0), ground_truth=image)
        with patch('database.export_images.DB_PATH', self.path):
            exported = export_images(Path(self.directory.name) / 'exports')
        rendered = render_run(run_hash(desc), Path(self.directory.name) / 'viewer')
        self.assertEqual(len(exported), 4)
        self.assertEqual(len(rendered), 2)
        from matplotlib.image import imread
        black = next(path for path in exported if '__best_reconstruction__0' in path.name)
        white = next(path for path in exported if '__best_reconstruction__1' in path.name)
        self.assertTrue(np.all(imread(black)[..., :3] == 0))
        self.assertTrue(np.all(imread(white)[..., :3] == 1))

    def test_default_setup_migrates_without_removing_results_and_reset_is_explicit(self):
        self.path.unlink()
        old_schema = SCHEMA.replace('    config_hash TEXT,\n', '').replace('    attempt INTEGER NOT NULL DEFAULT 1,\n', '').replace("    summary TEXT NOT NULL DEFAULT '{}',\n", '')
        with sqlite3.connect(self.path) as connection:
            connection.executescript(old_schema)
            connection.execute('INSERT INTO runs (run_hash,name,model,dataset,batch_size,attack_hash,defense_hash,raw_description) VALUES (?,?,?,?,?,?,?,?)',
                               ('legacy-original','legacy','m','d',1,'a','f','{}'))
            connection.execute('INSERT INTO logs (run_id,step,metrics) VALUES (1,1,?)', ('{"mse":1}',))
        setup_main([])
        with connection_module.get_connection() as connection:
            row = connection.execute('SELECT run_hash,config_hash,attempt,summary FROM runs').fetchone()
            self.assertEqual(row, ('legacy-original', None, 1, '{}'))
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM logs').fetchone()[0], 1)
        setup_main(['--reset'])
        with connection_module.get_connection() as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM runs').fetchone()[0], 0)


class HashAndHeatmapTests(unittest.TestCase):
    def test_canonical_nested_hashes_preserve_order_only_where_meaningful(self):
        left = [function_package('noise', {'a': 1, 'b': {'c': 2, 'd': 3}})]
        right = [function_package('noise', {'b': {'d': 3, 'c': 2}, 'a': 1})]
        self.assertEqual(config_hash(left), config_hash(right))
        self.assertTrue(config_hash(left).startswith('v2:'))
        self.assertNotEqual(config_hash(left + [function_package('other')]), config_hash([function_package('other')] + left))
        with self.assertRaises(TypeError):
            config_hash(object())
        with self.assertRaises(ValueError):
            config_hash({'bad': float('nan')})

    def test_heatmap_aggregates_and_rejects_mixed_protocols(self):
        rows = [('dlg', 'none', 'success'), ('dlg', 'none', 'failure'), ('dlg', 'none', 'error')]
        data = summarize_results(rows)
        self.assertEqual(data[2].item(), .5)
        self.assertEqual(data[3].item(), 2)
        self.assertEqual(data[4].item(), 1)
        np.testing.assert_array_equal(build_matrix(rows)[2], build_matrix(rows[::-1])[2])
        mixed = [dict(id=i, model=model, attack_name='dlg', defense_name='none', outcome='success')
                 for i, model in enumerate(['lenet5','resnet18'])]
        with self.assertRaisesRegex(ValueError, 'stratum'):
            build_matrix(mixed)

    def test_heatmap_latest_retry_is_one_sample(self):
        rows = [dict(id=i, config_hash='same', attempt=i, attack_name='dlg', defense_name='none', outcome=outcome)
                for i,outcome in ((1,'error'), (2,'success'))]
        data = summarize_results(rows)
        self.assertEqual(data[2].item(), 1)
        self.assertEqual(data[3].item(), 1)
        self.assertEqual(data[4].item(), 0)

    def test_heatmap_budget_changes_create_different_attack_configurations(self):
        rows = [dict(id=i, model='m', attack_name='dlg', attack_hash='same-attack', defense_name='none',
                     outcome='success', raw_description=json.dumps({'observer': {'fail_policies': [
                         {'name': 'max_iterations', 'parameters': {'limit': limit}}]}}))
                for i, limit in enumerate((10, 100))]
        attacks, _, matrix, counts, _ = summarize_results(rows)
        self.assertEqual(len(attacks), 2)
        self.assertEqual(matrix.shape, (2, 1))
        self.assertTrue(np.all(counts == 1))

    def test_heatmap_class_populations_are_canonical_and_separate(self):
        def row(index, classes):
            return dict(id=index, model='m', dataset='cifar10', batch_size=1,
                        attack_name='dlg', defense_name='none', outcome='success' if index == 0 else 'failure',
                        raw_description=json.dumps({'data': {'classes': classes}}))
        with self.assertRaisesRegex(ValueError, 'stratum'):
            build_matrix([row(0, [0]), row(1, [9])])
        _, _, matrix, counts, _ = summarize_results([row(0, [9, 0, 9]), row(1, [0, 9])])
        self.assertEqual(matrix.item(), .5)
        self.assertEqual(counts.item(), 2)

    def test_heatmap_separates_explicit_and_recorded_dataset_domains(self):
        for field, alternatives in (('split', ('train', 'test')), ('input_space', ('raw_pixels', 'normalized'))):
            rows = [dict(id=i, attack_name='dlg', defense_name='none', outcome='success',
                         raw_description=json.dumps({'data': {'classes': [0], field: value}}))
                    for i, value in enumerate(alternatives)]
            with self.subTest(location='data', field=field), self.assertRaisesRegex(ValueError, 'stratum'):
                build_matrix(rows)
        rows = [dict(id=i, attack_name='dlg', defense_name='none', outcome='success',
                     summary=json.dumps({'experiment_metadata': {'dataset_split': split, 'input_space': 'raw_pixels'}}))
                for i, split in enumerate(('train', 'test'))]
        with self.assertRaisesRegex(ValueError, 'stratum'):
            build_matrix(rows)


if __name__ == '__main__':
    unittest.main()
