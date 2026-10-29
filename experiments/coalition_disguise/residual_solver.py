"""Shorter, sparser residuals to a fixed gradient-nearest public disguise.

This is a restricted, audited search, not a global L0 optimizer. Projection
removes the part of each share orthogonal to all reference gradients. Cancelling
adjustments in that orthogonal space then match selected disguise coordinates.
In exact arithmetic the latter preserve every relative-distance inequality and
every nearest-disguise assignment. Float32 observations are audited afresh.
"""
import math

import torch

from .geometry import audit, observe, solve, validate_gradients


class ResidualSearchError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def reference_basis(g, d):
    matrix = torch.cat([g.double()[None], d.double()]).T
    q, r = torch.linalg.qr(matrix, mode='reduced')
    u, values, _ = torch.linalg.svd(r, full_matrices=False)
    threshold = max(matrix.shape) * torch.finfo(torch.float64).eps * float(values.max())
    return q @ u[:, values > threshold]


def project_shares(shares, g, basis):
    projected = (shares.double() @ basis) @ basis.T
    # Restore the tiny numerical projection residual; audit both precisions.
    projected[-1] = g.double() - projected[:-1].sum(0)
    return projected


def complement_lower_bound(g, d):
    """Float64 evaluation of min_jk ||G-D_j-D_k||² / 4.

    A necessary bound on the worst distance for exact aggregation. It does not
    prove feasibility, and is not used to excuse a failed numerical audit.
    """
    g, d = g.double(), d.double()
    norms, gd = d.square().sum(1), d @ g
    pairs = (g.square().sum() + norms[:, None] + norms[None, :]
             + 2 * (d @ d.T) - 2 * gd[:, None] - 2 * gd[None, :])
    j, k = divmod(int(pairs.argmin()), len(d))
    # Direct subtraction avoids cancellation in the reported value.
    return dict(complement_lower_bound_squared=float((g-d[j]-d[k]).square().sum()/4),
                complement_public_slots=[j, k])


def match_coordinates(projected, g, d, basis, rows, count, rng):
    """Fit disjoint coordinate sets, compensating on unused coordinates.

    Corrections are perpendicular to the reference span, including G. We reserve
    all matching sets up front, so later fits cannot disturb earlier matches.
    The free-coordinate solve finds the minimum-norm compensating correction.
    """
    p = len(g)
    available = torch.ones(p, dtype=torch.bool)
    matches = []
    # Different traversal orders explore different disjoint coordinate patterns.
    order = torch.randperm(len(rows), generator=rng).tolist()
    for index in order:
        row = rows[index]
        target = d[row['public_slot']]
        costs = (observe(projected, row['servers']) - target).abs()
        # Matching a coordinate where D == G cannot give a sparsity advantage.
        costs[~available | (target == g)] = torch.inf
        indices = torch.argsort(costs, stable=True)[:count]
        if len(indices) != count or not torch.isfinite(costs[indices]).all():
            return None
        available[indices] = False
        matches.append((index, indices))
    if int(available.sum()) < basis.shape[1]:
        return None
    free = basis[available]
    gram = free.T @ free
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    tolerance = max(gram.shape[0], 1) * torch.finfo(torch.float64).eps * 100
    if len(eigenvalues) and float(eigenvalues.min()) <= tolerance:
        return None  # This matching pattern leaves an unstable compensation solve.
    inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
    result = projected.clone()
    for index, indices in matches:
        row = rows[index]
        correction = torch.zeros_like(g)
        correction[indices] = d[row['public_slot'], indices] - observe(result, row['servers'])[indices]
        correction[available] = free @ (inverse @ (-(basis[indices].T @ correction[indices])))
        inside = row['servers'][0]
        outside = next(i for i in range(len(result)) if i not in row['servers'])
        result[inside] += correction
        result[outside] -= correction
    result[-1] = g - result[:-1].sum(0)
    return result, [dict(coalition=rows[i]['coalition'], coordinates=idx.tolist()) for i, idx in matches]


def solve_residual(g, d, n=3, seed=2026, relative_margin=.01, partitions=8, anchors=16, *,
                   residual_nnz_gap=1, max_disguise_distance_squared=None):
    """Return only shares passing aggregation, distance, and residual L0 checks.

    A gap of one requires strictly fewer nonzeros; zero permits equality. Search
    failures never relax that gap, the relative margin, or the absolute cap.
    """
    if not isinstance(residual_nnz_gap, int) or isinstance(residual_nnz_gap, bool) or residual_nnz_gap < 0:
        raise ValueError('Residual nonzero gap must be a nonnegative integer')
    if max_disguise_distance_squared is not None and (
            not math.isfinite(max_disguise_distance_squared) or max_disguise_distance_squared < 0):
        raise ValueError('Maximum squared disguise distance must be finite and nonnegative')
    # Retain the original construction as a reproducible initializer. Absolute
    # distances take priority over share sparsity when choosing its candidates.
    g, d = validate_gradients(g, d)
    basis = reference_basis(g, d)
    source = solve(g, d, n, seed, relative_margin, partitions, anchors, distance_first=True, score_basis=basis)
    projected = project_shares(source['exact_shares'], g, basis)
    _, rows = audit(projected.float(), g, d, source['epsilon'], projected)
    bounds = complement_lower_bound(g, d)
    attempts, skipped_patterns, best = [], [], None

    def consider(h, matching, name):
        nonlocal best
        if not torch.isfinite(h).all() or not torch.isfinite(h.float()).all():
            return
        report, geometry = audit(h.float(), g, d, source['epsilon'], h,
                                residual_nnz_gap=residual_nnz_gap,
                                max_disguise_distance_squared=max_disguise_distance_squared)
        attempts.append(dict(candidate=name, all_coalitions_pass=report['all_coalitions_pass'],
            float32_aggregation_pass=report['float32_aggregation_pass'],
            float64_aggregation_pass=report['float64_aggregation_pass'],
            minimum_residual_nnz_advantage=report['minimum_residual_nnz_advantage'],
            maximum_disguise_distance_squared=report['maximum_disguise_distance_squared'],
            coalition_geometry=geometry))
        if not (report['all_coalitions_pass'] and report['float32_aggregation_pass']
                and report['float64_aggregation_pass']):
            return
        score = (*report['ranked_disguise_distances_squared'],
                 *sorted((r['disguise_residual_nnz'] for r in geometry), reverse=True),
                 report['extra_nnz'])
        if best is None or score < best[0]:
            best = (score, dict(shares=h.float(), exact_shares=h, audit=report, geometry=geometry,
                               matched_coordinates=matching))

    consider(projected, [], 'projection')
    rng = torch.Generator().manual_seed(seed)
    # Leave enough unreserved coordinates for a full-rank compensation solve.
    capacity = max(0, (len(g)-basis.shape[1]-1)//len(rows))
    initial_count = min(capacity, max(8, 4*residual_nnz_gap))
    for iteration in range(partitions):
        count = min(capacity, initial_count * (2**iteration))
        if count < 1:
            break
        candidate = match_coordinates(projected, g, d, basis, rows, count, rng)
        if candidate is not None:
            h, matching = candidate
            consider(h, matching, f'matching_{iteration}_count_{count}')
        else:
            skipped_patterns.append(dict(iteration=iteration, coordinates_per_subset=count,
                reason='Insufficient eligible/free coordinates or unstable compensation solve'))
    diagnostics = dict(solver='residual_sparse_v1', reference_span_dimension=basis.shape[1],
        residual_nnz_gap=residual_nnz_gap, max_disguise_distance_squared=max_disguise_distance_squared,
        sparsity_measure='exact nonzero count of float64 differences of float32 observations and reference gradients',
        objective='lexicographic descending squared disguise distances, then descending disguise residual nonzeros, then extra share nonzeros',
        search_budget=partitions, searched_candidates=len(attempts), attempts=attempts,
        skipped_patterns=skipped_patterns, initializer_geometry=source['geometry'], **bounds)
    if best is None:
        raise ResidualSearchError(
            'No decomposition passed aggregation, distance, and residual sparsity checks within the search budget; '
            'thresholds were not relaxed. This is not an infeasibility proof.', diagnostics)
    result = best[1]
    result.update(epsilon=source['epsilon'], scale=source['scale'], residual_nnz_gap=residual_nnz_gap,
        max_disguise_distance_squared=max_disguise_distance_squared, solver='residual_sparse_v1',
        search_diagnostics=diagnostics, initial_shares=source['shares'],
        initial_exact_shares=source['exact_shares'], owners=source['owners'],
        repair_coordinate=source['repair_coordinate'], alpha=source['alpha'], magnitude=source['magnitude'])
    return result
