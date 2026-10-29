"""Sparse cancelling repair; proper coalitions always observe raw sums."""
import itertools
import math
import torch


def coalitions(n):
    if not 2 <= n <= 8:
        raise ValueError("Exhaustive verification supports 2..8 servers")
    return [tuple(i for i in range(n) if mask & (1 << i))
            for mask in range(1, (1 << n) - 1)]


def observe(shares, subset):
    # Same float32 reduction is used for audits and attacker observations.
    return shares[list(subset)].sum(dim=0)


def margins(observations, g, d):
    observations, g, d = observations.double(), g.double(), d.double()
    # Distance difference cancels the quadratic observation term.
    return 2 * observations @ (d - g).T + g.square().sum() - d.square().sum(1)


def audit(shares, g, d, epsilon, exact_shares=None, *, residual_nnz_gap=None,
          max_disguise_distance_squared=None):
    """Audit the actual float32 observations, including exact residual nonzeros.

    No tolerance or rounding-to-zero is used for sparsity. Historical sparse
    decompositions report the counts but do not require residual sparsity.
    """
    subsets = coalitions(len(shares))
    g, d = g.double(), d.double()
    rows = []
    for subset in subsets:
        h = observe(shares.float(), subset).double()
        private_residual, public_residuals = h - g, d - h
        private_distance = private_residual.square().sum()
        distances = public_residuals.square().sum(1)
        nearest = int(distances.argmin())
        margin = float(private_distance - distances[nearest])
        private_nnz = int(torch.count_nonzero(private_residual))
        public_nnz = int(torch.count_nonzero(public_residuals[nearest]))
        simpler = public_nnz + (residual_nnz_gap or 0) <= private_nnz
        closer = margin >= epsilon
        absolute = (max_disguise_distance_squared is None
                    or float(distances[nearest]) <= max_disguise_distance_squared)
        rows.append(dict(coalition=sum(1 << i for i in subset), servers=list(subset),
                         private_distance=float(private_distance),
                         public_distance=float(distances[nearest]), public_slot=nearest,
                         private_norm=float(private_distance.sqrt()),
                         public_norm=float(distances[nearest].sqrt()),
                         disguise_residual_nnz=public_nnz, ground_truth_residual_nnz=private_nnz,
                         residual_nnz_advantage=private_nnz-public_nnz,
                         residual_sparsity_pass=simpler, relative_distance_pass=closer,
                         absolute_distance_pass=absolute, margin=margin,
                         passed=closer and absolute and (residual_nnz_gap is None or simpler)))
    aggregate = shares.float().sum(0).double()
    error = aggregate - g.double()
    overlap = (shares != 0).sum(0) > 1
    report = dict(epsilon=epsilon, all_coalitions_pass=all(r['passed'] for r in rows),
                  residual_sparsity_required=residual_nnz_gap is not None,
                  residual_nnz_gap=residual_nnz_gap,
                  all_residual_sparsity_pass=all(r['residual_sparsity_pass'] for r in rows),
                  minimum_residual_nnz_advantage=min(r['residual_nnz_advantage'] for r in rows),
                  all_relative_distances_pass=all(r['relative_distance_pass'] for r in rows),
                  max_disguise_distance_squared=max_disguise_distance_squared,
                  all_absolute_distances_pass=all(r['absolute_distance_pass'] for r in rows),
                  ranked_disguise_distances_squared=sorted((r['public_distance'] for r in rows), reverse=True),
                  maximum_disguise_distance_squared=max(r['public_distance'] for r in rows),
                  mean_disguise_distance_squared=sum(r['public_distance'] for r in rows)/len(rows),
                  float32_max_error=float(error.abs().max()),
                  float32_relative_error=float(error.norm() / g.double().norm().clamp_min(1e-30)),
                  float32_aggregation_pass=bool(torch.allclose(aggregate, g.double(), rtol=1e-5, atol=1e-6)),
                  extra_nnz=int((shares != 0).sum() - (g != 0).sum()),
                  overlap_coordinates=overlap.nonzero().flatten().tolist(),
                  per_server_nnz=(shares != 0).sum(1).tolist(),
                  per_server_norm=shares.double().norm(dim=1).tolist(),
                  maximum_share_magnitude=float(shares.abs().max()),
                  norm_amplification=float(shares.double().norm(dim=1).max() / g.double().norm().clamp_min(1e-30)))
    if exact_shares is not None:
        report['float64_max_error'] = float((exact_shares.sum(0) - g.double()).abs().max())
        report['float64_aggregation_pass'] = bool(torch.allclose(exact_shares.sum(0), g.double(), rtol=1e-12, atol=1e-12))
    return report, rows


def candidate_score(report, rows, distance_first=False):
    if distance_first:
        # Worst absolute distance, then second-worst, etc. Sparsity breaks ties.
        return (*sorted((r['public_distance'] for r in rows), reverse=True), report['extra_nnz'])
    return (report['extra_nnz'], max(r['public_distance'] for r in rows),
            sum(r['public_distance'] for r in rows))


def validate_gradients(g, d):
    g, d = g.double(), d.double()
    if g.ndim != 1 or not len(g) or d.ndim != 2 or d.shape[1] != len(g) or not len(d):
        raise ValueError('Expected one gradient vector and a nonempty public gradient matrix')
    if not torch.isfinite(g).all() or not torch.isfinite(d).all():
        raise ValueError('Gradients must be finite')
    return g, d


def solve(g, d, n=3, seed=2026, relative_margin=.01, partitions=8, anchors=16, *,
          distance_first=False, max_disguise_distance_squared=None, score_basis=None):
    g, d = validate_gradients(g, d)
    if not math.isfinite(relative_margin) or relative_margin < 0 or partitions < 1 or anchors < 1:
        raise ValueError('Invalid search budget or relative margin')
    if max_disguise_distance_squared is not None and (
            not math.isfinite(max_disguise_distance_squared) or max_disguise_distance_squared < 0):
        raise ValueError('Maximum squared disguise distance must be finite and nonnegative')
    subsets = coalitions(n)
    scale = float((d - g).square().sum(1).median())
    epsilon = relative_margin * scale
    # A small extra search margin protects float32 serialization.
    target = epsilon + max(scale, 1.) * 1e-5
    delta = d - g
    leverage = torch.minimum(delta.max(0).values, -delta.min(0).values)
    coordinates = torch.argsort(leverage, descending=True)[:anchors]
    coordinates = [int(k) for k in coordinates if leverage[k] > 0]
    rng = torch.Generator().manual_seed(seed)
    public_coefficients = None if score_basis is None else d @ score_basis
    best = None
    for _ in range(partitions):
        owners = torch.randint(n, (len(g),), generator=rng)
        base = torch.stack([g * (owners == i) for i in range(n)])
        base_coefficients = None if score_basis is None else base @ score_basis
        observations = torch.stack([observe(base, s) for s in subsets])
        initial = margins(observations, g, d)
        candidates = [(None, torch.zeros(n, dtype=torch.float64), 0.)]
        if not bool((initial.max(1).values >= epsilon).all()):
            candidates = []
            for k in coordinates:
                for size in range(2, n + 1):
                    for active in itertools.combinations(range(n), size):
                        for negative in active:
                            alpha = torch.zeros(n, dtype=torch.float64)
                            alpha[list(active)] = 1
                            alpha[negative] = -(size - 1)
                            coefficients = torch.tensor([alpha[list(s)].sum() for s in subsets])
                            slopes = 2 * coefficients[:, None] * delta[:, k][None, :]
                            # Each coalition accepts a union of half-lines in t.
                            # Feasibility can change only at a nonnegative boundary.
                            nonzero = slopes != 0
                            roots = ((target - initial)[nonzero] / slopes[nonzero])
                            times = torch.unique(torch.cat([torch.zeros(1), roots[roots >= 0]]))
                            for t in times.tolist():
                                if bool(((initial + t * slopes).max(1).values >= target - 1e-10).all()):
                                    candidates.append((k, alpha, t))
                                    break
        for k, alpha, t in candidates:
            h = base.clone()
            if k is not None:
                h[:, k] += t * alpha
            h32 = h.float()
            score = None
            if score_basis is not None:
                # Screen in the small reference space before the full audit.
                # This score is also used after auditing, so no potentially
                # better candidate is pruned by an approximate surrogate.
                coefficients = base_coefficients.clone()
                if k is not None:
                    coefficients += t * alpha[:, None] * score_basis[k]
                small_observations = torch.stack([observe(coefficients, s) for s in subsets])
                distances = (small_observations[:, None]-public_coefficients).square().sum(2).min(1).values
                scoring_rows = [dict(public_distance=float(value)) for value in distances]
                score = candidate_score(dict(extra_nnz=int((h32 != 0).sum()-(g != 0).sum())),
                                        scoring_rows, distance_first)
                if best is not None and score >= best[0]:
                    continue
            report, rows = audit(h32, g, d, epsilon, h,
                                 max_disguise_distance_squared=max_disguise_distance_squared)
            if not report['all_coalitions_pass'] or not report['float32_aggregation_pass']:
                continue
            if score is None:
                score = candidate_score(report, rows, distance_first)
            if best is None or score < best[0]:
                best = (score, dict(shares=h32, exact_shares=h, owners=owners,
                                   repair_coordinate=k, alpha=alpha, magnitude=t,
                                   epsilon=epsilon, scale=scale, audit=report, geometry=rows))
        if not distance_first and best is not None and best[0][0] == 0:
            break
    if best is None:
        raise RuntimeError('No feasible decomposition within search budget; this is not an infeasibility proof')
    return best[1]
