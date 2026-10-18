"""Comparable-stratum reconstruction success rates, including sample counts."""
import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from database.connection import DB_PATH
from database.hashing import config_hash


def load_results():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute('SELECT * FROM runs ORDER BY id')]


def stratum_key(row):
    description = row.get('raw_description', '{}')
    description = json.loads(description) if isinstance(description, str) else description
    config = description.get('observer') or {}
    data_config = dict(description.get('data') or {})
    classes = data_config.get('classes')
    # Class ordering and duplicate entries do not change the sampled population.
    data_config['classes'] = sorted(set(classes)) if classes is not None else None
    summary = row.get('summary', '{}')
    summary = json.loads(summary) if isinstance(summary, str) else (summary or {})
    metadata = summary.get('experiment_metadata') or {}
    dataset_context = {
        'data': data_config,
        'dataset_split': description.get('dataset_split', metadata.get('dataset_split')),
        'input_space': description.get('input_space', metadata.get('input_space')),
    }
    target = config.get('evaluation_target_mse')
    # Legacy outcomes depended on explicit oracle success policies.
    policies = config.get('success_policies') or []
    return (row.get('model'), row.get('dataset'), row.get('batch_size'),
            json.dumps(dataset_context, sort_keys=True),
            description.get('protocol_version', 'legacy'), config.get('batch_matching', 'ordered'),
            target, json.dumps(policies, sort_keys=True),
            json.dumps(config.get('best_selection'), sort_keys=True),
            json.dumps(config.get('oracle_selection'), sort_keys=True))


def _rows(rows):
    converted = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            converted.append(row)
        else:
            attack, defense, outcome = row
            converted.append(dict(id=index, attack_name=attack, defense_name=defense, outcome=outcome))
    # A retry is another attempt at one experiment, not another independent sample.
    latest = {}
    for row in converted:
        identity = row.get('config_hash') or row.get('run_hash') or ('row', row.get('id'))
        previous = latest.get(identity)
        if previous is None or (row.get('attempt', 1), row.get('id', 0)) > (previous.get('attempt', 1), previous.get('id', 0)):
            latest[identity] = row
    return list(latest.values())


def summarize_results(rows, stratum=None):
    rows = _rows(rows)
    strata = {stratum_key(row) for row in rows}
    if stratum is None and len(strata) > 1:
        raise ValueError('Results contain different models, datasets, class populations, input domains, batch sizes or evaluation protocols; select one stratum')
    if stratum is not None:
        rows = [row for row in rows if stratum_key(row) == stratum]
    def label(row, kind):
        name = row.get(kind + '_name') or 'unknown'
        identity = row.get(kind + '_hash')
        if kind == 'attack' and row.get('raw_description'):
            description = row['raw_description']
            description = json.loads(description) if isinstance(description, str) else description
            # Iteration/patience budgets live on the observer, outside attack_hash.
            identity = config_hash({'attack': identity, 'fail_policies': (description.get('observer') or {}).get('fail_policies')})
        return f'{name} [{identity}]' if identity else name
    attacks = sorted({label(row, 'attack') for row in rows})
    defenses = sorted({label(row, 'defense') for row in rows})
    count = np.zeros((len(attacks), len(defenses)), dtype=int)
    success = np.zeros_like(count)
    excluded = np.zeros_like(count)
    for row in rows:
        cell = (attacks.index(label(row, 'attack')), defenses.index(label(row, 'defense')))
        if row.get('outcome') in {'success', 'failure'}:
            count[cell] += 1
            success[cell] += row['outcome'] == 'success'
        else:
            excluded[cell] += 1
    matrix = np.divide(success, count, out=np.full(count.shape, np.nan), where=count != 0)
    return attacks, defenses, matrix, count, excluded


def build_matrix(rows, stratum=None):
    return summarize_results(rows, stratum)[:3]


def plot_heatmap(attacks, defenses, matrix, counts=None, excluded=None, output='attack_defense_heatmap.png', title=None):
    if not attacks or not defenses:
        raise ValueError('No runs are available for this heatmap')
    fig, ax = plt.subplots(figsize=(max(8, len(defenses) * 2), max(4, len(attacks))))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect='auto')
    # Full identities distinguish configurations; short labels keep plots readable.
    short = lambda label: label if len(label) < 70 else label.split(' [')[0] + ' [' + label.split(' [')[1][:15] + '…]'
    ax.set_xticks(range(len(defenses)), [short(label) for label in defenses], rotation=60, ha='right')
    ax.set_yticks(range(len(attacks)), [short(label) for label in attacks])
    ax.set_xlabel('Defense configuration')
    ax.set_ylabel('Attack configuration')
    ax.set_title(title or 'Reconstruction success rate (completed experiments)')
    for row in range(len(attacks)):
        for column in range(len(defenses)):
            value = matrix[row, column]
            label = '-' if np.isnan(value) else f'{value:.0%}'
            if counts is not None:
                label += f'\nn={counts[row, column]}'
            if excluded is not None and excluded[row, column]:
                label += f'\nexcluded={excluded[row, column]}'
            ax.text(column, row, label, ha='center', va='center')
    fig.colorbar(image, ax=ax, label='Success fraction')
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    args = parser.parse_args(argv)
    rows = load_results()
    groups = defaultdict(list)
    for row in rows:
        groups[stratum_key(row)].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (stratum, group) in enumerate(sorted(groups.items(), key=lambda item: repr(item[0]))):
        data = summarize_results(group)
        destination = args.output_dir / f'attack_defense_heatmap_{index}.png'
        plot_heatmap(*data[:3], counts=data[3], excluded=data[4], output=destination, title=str(stratum))
        print(f'{destination}: stratum={stratum!r}')


if __name__ == '__main__':
    main()
