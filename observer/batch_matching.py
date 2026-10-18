"""Permutation-invariant evaluation of reconstructed image batches."""
import torch
import math


def minimum_assignment(cost):
    """Return the minimum-cost column for each row (square Hungarian method)."""
    n = len(cost)
    if any(len(row) != n for row in cost):
        raise ValueError("Assignment costs must be square")
    if any(not math.isfinite(value) for row in cost for value in row):
        raise ValueError("Assignment costs must be finite")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [float('inf')] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, n + 1):
                if not used[j]:
                    candidate = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if candidate < minimum[j]:
                        minimum[j], way[j] = candidate, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def align_batch(reconstruction, ground_truth, mode="assignment"):
    """Align samples by minimum total pixel MSE, preserving the raw tensors.

    Indices map each true sample to its original reconstructed sample. This is
    an evaluation operation and supplies no information to the attack.
    """
    if mode not in {"assignment", "ordered"}:
        raise ValueError("batch_matching must be 'assignment' or 'ordered'")
    if not isinstance(reconstruction, torch.Tensor) or not isinstance(ground_truth, torch.Tensor):
        return reconstruction, None
    if reconstruction.shape != ground_truth.shape:
        raise ValueError("Reconstruction and ground truth shapes must match")
    if reconstruction.ndim != 4 or reconstruction.shape[0] <= 1 or mode == "ordered":
        return reconstruction, list(range(reconstruction.shape[0])) if reconstruction.ndim == 4 else None
    source = reconstruction.detach().reshape(len(reconstruction), -1).to(device='cpu', dtype=torch.float64)
    target = ground_truth.detach().reshape(len(ground_truth), -1).to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError("Batch matching requires finite images")
    costs = torch.stack([((source - sample) ** 2).mean(dim=1) for sample in target])
    if not torch.isfinite(costs).all():
        raise ValueError("Batch assignment costs are not finite")
    assignment = minimum_assignment(costs.tolist())
    return reconstruction[assignment], assignment
