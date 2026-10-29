"""Serialization and gradient conventions shared by the experiment tools."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = f'{tensor.dtype}:{list(tensor.shape)}:'.encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


def model_digest(model):
    return hashlib.sha256(json.dumps([
        (name, tensor_digest(value)) for name, value in model.state_dict().items()
    ]).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gradient(model, image, label):
    # Integer labels and explicitly supplied soft-label probabilities use the
    # same CE convention as the registered reconstruction engine.
    loss = F.cross_entropy(model(image), label)
    return torch.cat([g.detach().flatten() for g in torch.autograd.grad(loss, tuple(model.parameters()))])


def unpack(vector, model):
    result, offset = [], 0
    for p in model.parameters():
        result.append(vector[offset:offset+p.numel()].reshape_as(p).clone())
        offset += p.numel()
    if offset != len(vector):
        raise ValueError('Gradient parameter ordering/size mismatch')
    return result


def pixel_metrics(image):
    return dict(pixel_min=float(image.min()), pixel_max=float(image.max()),
                outside_pixel_fraction=float(((image < 0) | (image > 1)).float().mean()))
