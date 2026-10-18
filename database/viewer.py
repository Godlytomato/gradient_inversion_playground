"""Render stored raw-domain images with their saved evaluation assignment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from database.connection import get_connection
from database.database import get_run_images
from database.image_display import viewable, align_for_display


def get_run_id(run_hash):
    connection = get_connection()
    try:
        row = connection.execute('SELECT id FROM runs WHERE run_hash = ?', (run_hash,)).fetchone()
    finally:
        connection.close()
    if row is None:
        raise KeyError(f'Unknown run hash: {run_hash}')
    return row[0]


def render_run(run_hash, output_dir=Path('temp_view')):
    run_id = get_run_id(run_hash)
    images = get_run_images(run_id)
    if 'ground_truth' not in images or 'best_reconstruction' not in images:
        raise KeyError('Run needs ground_truth and best_reconstruction images')
    connection = get_connection()
    try:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(runs)')}
        summary = json.loads(connection.execute('SELECT summary FROM runs WHERE id = ?', (run_id,)).fetchone()[0]) if 'summary' in columns else {}
    finally:
        connection.close()
    kinds = ['ground_truth', 'best_reconstruction']
    if 'oracle_reconstruction' in images:
        kinds.append('oracle_reconstruction')
    images = {kind: align_for_display(images[kind], kind, summary) for kind in kinds}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(len(images['ground_truth'])):
        figure, axes = plt.subplots(1, len(kinds), squeeze=False)
        for axis, kind in zip(axes[0], kinds):
            image = viewable(images[kind][index])
            axis.imshow(image, cmap='gray' if image.ndim == 2 else None, vmin=0, vmax=1)
            axis.set_title({'ground_truth': 'Original', 'best_reconstruction': 'Attack-selected', 'oracle_reconstruction': 'Oracle (evaluation only)'}[kind])
            axis.axis('off')
        figure.tight_layout()
        path = output_dir / f'run_{run_id}_{index}.png'
        figure.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(figure)
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_hash')
    parser.add_argument('--output-dir', type=Path, default=Path('temp_view'))
    args = parser.parse_args(argv)
    for path in render_run(args.run_hash, args.output_dir):
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
