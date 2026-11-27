"""Export completed direction-sweep results and print a compact summary."""
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from database.connection import DB_PATH


def main():
    with sqlite3.connect(f'file:{DB_PATH}?mode=ro', uri=True) as con:
        rows = con.execute('''
            SELECT r.id, r.name, r.attack_name, r.raw_description,
                   r.run_metrics, MIN(CAST(json_extract(l.metrics, '$.mse') AS REAL))
            FROM runs r JOIN logs l ON l.run_id = r.id
            WHERE r.name LIKE 'svd_direction_v1__%'
              AND r.outcome IN ('success', 'failure')
            GROUP BY r.id ORDER BY r.name
        ''').fetchall()
    fields = ['run_id', 'name', 'attack', 'direction', 'percent', 'configured_delta_l2', 'actual_delta_l2', 'best_mse', 'defense_success']
    groups = defaultdict(list)
    count = 0
    with Path('svd_direction_results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for run_id, name, attack, raw, metrics, mse in rows:
            if mse is None:
                continue
            config = json.loads(raw)
            if config.get('defense_name') != 'indexed_svd_noise':
                continue
            t = config['gradient_transformations'][0]['transform']['parameters']
            direction = t['construction']['parameters']['direction']
            scale = t['scaling']['parameters']['scale']
            percent = int(name.split('__')[-2][1:])
            delta = json.loads(metrics or '{}').get('delta_l2')
            threshold = config['observer'].get('evaluation_target_mse')
            if threshold is None:
                threshold = 0.02
            success = mse > threshold
            w.writerow([run_id, name, attack, direction, percent, scale, delta, mse, success])
            groups[(name.split('__')[-1], direction, percent)].append((mse, success))
            count += 1
    print('mode  direction  percent  mean best MSE  blocked/total')
    for (mode, direction, percent), values in sorted(groups.items()):
        print(f'{mode:5} {direction:9} {percent:8} {sum(v[0] for v in values)/len(values):14.7g} {sum(v[1] for v in values):3}/{len(values)}')
    print(f'Saved {count} rows to svd_direction_results.csv')


if __name__ == '__main__':
    main()
