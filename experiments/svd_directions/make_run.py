"""Generate configs from recorded max-SVD lower-bound reference runs."""
import argparse
import copy
import csv
import json
import math
import random
import sqlite3
from dataclasses import asdict
from pathlib import Path

import yaml

from database.connection import DB_PATH
from validation.implementations.defenses.indexed_svd_noise import indexed_svd_noise

ATTACKS = ('beyond_gradients', 'gi_pip', 'hf_gradinv')
MODEL = 'resnet18'
RUN_TAG = 'svd_direction_v1'
INPUT_DIM = 3 * 32 * 32
DIRECTIONS = [1, *sorted(random.Random(42).sample(range(2, INPUT_DIM), 5)), -1]
PERCENTAGES = range(10, 101, 10)


def load_references():
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as con:
        rows = con.execute('''
            SELECT id, name, attack_name, raw_description, run_metrics
            FROM runs
            WHERE model = ? AND dataset = 'cifar10' AND batch_size = 1
              AND outcome IN ('success', 'failure')
              AND (defense_name = 'max_svd_lower_bound' OR defense_name IS NULL)
            ORDER BY id DESC
        ''', (MODEL,)).fetchall()
    references = {}
    for run_id, name, attack, raw, metrics in rows:
        config = json.loads(raw)
        transforms = config.get('gradient_transformations', [])
        defense = config.get('defense_name') or (transforms[0]['name'] if len(transforms) == 1 else '')
        attack = attack or config.get('attack_name') or name.split('__')[0]
        if defense != 'max_svd_lower_bound' or attack not in ATTACKS or attack in references:
            continue
        delta = json.loads(metrics or '{}').get('delta_l2')
        if delta is None or not math.isfinite(float(delta)) or float(delta) <= 0:
            continue
        references[attack] = (run_id, config, float(delta))
    missing = set(ATTACKS) - references.keys()
    if missing:
        raise RuntimeError(f'No usable max-SVD delta_l2 for {sorted(missing)}')
    return references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('schedule'))
    parser.add_argument('--keep-early-stop', action='store_true',
                        help='retain the original observer stopping policies')
    args = parser.parse_args()
    refs = load_references()
    configs = []
    manifest = []
    for attack in ATTACKS:
        run_id, original, delta = refs[attack]
        print(f'{attack}: reference run {run_id}, delta_l2={delta:.10g}')
        for direction in DIRECTIONS:
            for percent in PERCENTAGES:
                scale = delta * percent / 100
                config = copy.deepcopy(original)
                rank = 'min' if direction == -1 else str(direction)
                mode = 'early' if args.keep_early_stop else 'fixed'
                config['name'] = f'{RUN_TAG}__{attack}__{MODEL}__r{run_id}__dir{rank}__p{percent:03d}__{mode}'
                config['attack_name'] = attack
                config['defense_name'] = 'indexed_svd_noise'
                config['attack']['base_seed'] = 0
                config['gradient_transformations'] = [asdict(t) for t in indexed_svd_noise(direction, scale)]
                observer = config['observer']
                if not args.keep_early_stop:
                    observer['success_policies'] = []
                    observer['fail_policies'] = [p for p in observer['fail_policies'] if p['name'] == 'max_iterations']
                    if not observer['fail_policies']:
                        raise RuntimeError(f'{attack}: no max_iterations policy; set a fixed budget first')
                if observer['logging']['logging_type'] != 'database':
                    raise RuntimeError('Reference must use database logging')
                if 'mse' not in observer['logging']['metrics']:
                    raise RuntimeError('Reference must log MSE')
                configs.append(config)
                manifest.append((config['name'], attack, run_id, direction, percent, delta, scale))
    args.output.mkdir(parents=True, exist_ok=True)
    for config in configs:
        (args.output / (config['name'] + '.yaml')).write_text(yaml.safe_dump(config, sort_keys=False))
    with open('svd_direction_manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'attack', 'reference_run_id', 'direction', 'percent', 'reference_delta_l2', 'scale'])
        w.writerows(manifest)
    print('Directions:', DIRECTIONS)
    print(f'Generated {len(configs)} configs in {args.output}')


if __name__ == '__main__':
    main()
