import io
import sqlite3
from pathlib import Path

import clip
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, ToPILImage

DB_PATH = Path('database/playground.db')
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_tensor(blob):
    image = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    return image.unsqueeze(0) if image.ndim == 3 else image


def pixel_key(image):
    return (image.clamp(0, 1) * 255).round().byte().numpy().tobytes()


def main():
    if not DB_PATH.is_file():
        raise FileNotFoundError(f'Database not found: {DB_PATH}')

    # Recover the labels of original images without modifying the experiments.
    dataset = CIFAR10(root='data', train=False, download=True, transform=ToTensor())
    ground_truth_labels = {}
    for image, label in dataset:
        key = pixel_key(image)
        if key in ground_truth_labels and ground_truth_labels[key] != label:
            raise RuntimeError('Ambiguous CIFAR-10 ground-truth image')
        ground_truth_labels[key] = label

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()
    prompts = [f'a photo of a {name}' for name in CLASSES]
    with torch.inference_mode():
        text_features = model.encode_text(clip.tokenize(prompts).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    with sqlite3.connect(DB_PATH) as db:
        backup = DB_PATH.with_suffix('.before_classification.db')
        if not backup.exists():
            with sqlite3.connect(backup) as output:
                db.backup(output)
            print('Created backup:', backup)
        else:
            print('Existing backup retained:', backup)

        columns = {row[1] for row in db.execute('PRAGMA table_info(runs)')}
        if 'best_image_classified' not in columns:
            db.execute('ALTER TABLE runs ADD COLUMN best_image_classified INTEGER')
        if 'best_image_classified_confidence' not in columns:
            db.execute('ALTER TABLE runs ADD COLUMN best_image_classified_confidence REAL')

        rows = db.execute('''
            SELECT r.id, r.batch_size, b.image_data, g.image_data
            FROM runs AS r
            JOIN images AS b ON b.image_hash = r.best_reconstruction_hash
            JOIN images AS g ON g.run_id = r.id AND g.image_type = 'ground_truth'
            WHERE r.dataset = 'cifar10' AND r.best_image_classified IS NULL
        ''').fetchall()
        converted = ToPILImage()
        completed = 0
        for run_id, batch_size, best_blob, gt_blob in rows:
            if batch_size != 1:
                print(f'Skipping {run_id}: batch size not 1')
                continue
            best, original = load_tensor(best_blob), load_tensor(gt_blob)
            if best.shape != (1, 3, 32, 32) or original.shape != (1, 3, 32, 32):
                print(f'Skipping {run_id}: unexpected image shape')
                continue
            if not torch.isfinite(best).all():
                print(f'Skipping {run_id}: nonfinite reconstruction')
                continue
            label = ground_truth_labels.get(pixel_key(original[0]))
            if label is None:
                print(f'Skipping {run_id}: original not found in CIFAR-10 test set')
                continue

            x = preprocess(converted(best[0].clamp(0, 1).float())).unsqueeze(0).to(device)
            with torch.inference_mode():
                image_features = model.encode_image(x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                scores = 100.0 * image_features @ text_features.T
                probabilities = scores.float().softmax(dim=-1)[0]
                predicted = int(probabilities.argmax().item())
                confidence = float(probabilities[predicted].item())
            correct = int(predicted == label)
            db.execute('''
                UPDATE runs
                SET best_image_classified = ?, best_image_classified_confidence = ?
                WHERE id = ?
            ''', (correct, confidence, run_id))
            print(f'Run {run_id}: truth={CLASSES[label]}, predicted={CLASSES[predicted]}, '
                  f'correct={bool(correct)}, confidence={confidence:.3f}')
            completed += 1
        db.commit()
        print(f'Done: classified {completed} run(s).')


if __name__ == '__main__':
    main()