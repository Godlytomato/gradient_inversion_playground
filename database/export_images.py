"""Export stored image tensors without changing their contrast or colors."""
import argparse
import io
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from database.connection import DB_PATH
from database.image_display import viewable, align_for_display

OUTPUT_DIR = Path('exported_images')


def save_tensor_image(image, output_path):
    image = viewable(image)
    plt.imsave(output_path, image, cmap='gray' if image.ndim == 2 else None, vmin=0, vmax=1)


def export_images(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        columns = {row[1] for row in connection.execute('PRAGMA table_info(runs)')}
        summary = 'runs.summary' if 'summary' in columns else "'{}'"
        rows = connection.execute(f'''SELECT runs.id, images.image_type, images.image_data, {summary}
                                     FROM images JOIN runs ON images.run_id = runs.id
                                     ORDER BY runs.id, images.image_type''').fetchall()
    paths = []
    for run_id, image_type, data, raw_summary in rows:
        tensor = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        tensor = align_for_display(tensor, image_type, json.loads(raw_summary))
        samples = list(tensor) if tensor.ndim == 4 else [tensor]
        for index, sample in enumerate(samples):
            path = output_dir / f'run_{run_id}__{image_type}__{index}.png'
            save_tensor_image(sample, path)
            paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    for path in export_images(args.output_dir):
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
