"""Unit noise in an indexed, nonzero left singular direction of J_g."""
import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from transformations.ranking_implimentations.transform_utils import (
    batch_tensors, jacobian_vector_product, jtj_vector_product,
)
from transformations.transformation_implimentations.tranform_utils import (
    vector_to_noise,
)

_CACHE = {}
_CACHE_DIR = Path('.svd_direction_cache')


def _fingerprint(context, images, labels):
    h = hashlib.sha256(b'indexed-svd-gram-fp64-v1')
    h.update(repr(context.model).encode())
    h.update(repr(context.loss_function).encode())
    for name, t in context.model.state_dict().items():
        h.update(name.encode())
        _hash_tensor(h, t)
    _hash_tensor(h, images)
    _hash_tensor(h, labels)
    return h.hexdigest()


def _hash_tensor(h, t):
    t = t.detach().cpu().contiguous()
    h.update(str((tuple(t.shape), t.dtype)).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())


def _spectrum(context):
    images, labels = batch_tensors(context)
    key = _fingerprint(context, images, labels)
    if key in _CACHE:
        return _CACHE[key]

    _CACHE_DIR.mkdir(exist_ok=True)
    path = _CACHE_DIR / f'{key}.pt'
    if path.exists():
        values, vectors = torch.load(path, map_location='cpu', weights_only=True)
    else:
        # Use double precision: eigenvalues of J^T J square the singular values.
        model = copy.deepcopy(context.model).double()
        batch = [(x.detach().double(), int(y)) for x, y in zip(images, labels)]
        ctx = replace(context, model=model, batch=batch)
        n = images.numel()
        gram = torch.empty((n, n), dtype=torch.float64, device='cpu')
        e = torch.zeros(n, dtype=torch.float64, device=images.device)
        print(f'Computing input Gram matrix: {n} columns', flush=True)
        for i in range(n):
            e[i] = 1
            gram[:, i] = jtj_vector_product(e, ctx).detach().cpu()
            e[i] = 0
            if (i + 1) % 64 == 0 or i + 1 == n:
                print(f'Gram: {i + 1}/{n}', flush=True)
        if not torch.isfinite(gram).all():
            raise RuntimeError('Nonfinite input Gram matrix')
        gram = (gram + gram.T) * 0.5
        values, vectors = torch.linalg.eigh(gram)
        values = values.flip(0).contiguous()
        vectors = vectors.flip(1).contiguous()
        tmp = path.with_suffix('.tmp')
        torch.save((values, vectors), tmp)
        tmp.replace(path)
        del model, gram, ctx
    _CACHE.clear()  # keep one model/image spectrum in RAM
    _CACHE[key] = (values, vectors)
    return values, vectors


def indexed_svd_alignment(package, selection_masks, context, direction: int = 1):
    """1 = largest, 2 = second largest, -1 = smallest resolved nonzero."""
    if direction == 0:
        raise ValueError('Direction must be nonzero (one-based indexing)')
    if not all(bool(mask.all()) for mask in selection_masks):
        raise ValueError('Use full support to preserve singular-vector alignment')
    values, vectors = _spectrum(context)
    # Do not select numerical null-space vectors: Jv cannot normalize to u.
    if not torch.isfinite(values).all() or values[0] <= 0:
        raise RuntimeError('Invalid Jacobian spectrum')
    rank = int((values > values[0] * 1e-12).sum().item())
    index = direction - 1 if direction > 0 else rank + direction
    if not 0 <= index < rank:
        raise ValueError(f'Direction {direction} outside resolved rank {rank}')
    v = vectors[:, index].clone()
    # SVD signs are arbitrary; choose a repeatable sign convention.
    if v[v.abs().argmax()] < 0:
        v.neg_()
    u = jacobian_vector_product(v, context).detach()
    norm = u.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError('Cannot normalize singular direction')
    print(f'SVD direction={direction}, resolved_rank={rank}, '
          f'sigma={values[index].sqrt().item():.8g}', flush=True)
    return vector_to_noise(package, selection_masks, u / norm)
