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


def audit(shares, g, d, epsilon, exact_shares=None):
    subsets = coalitions(len(shares))
    rows = []
    for subset in subsets:
        h = observe(shares.float(), subset).double()
        private_distance = (h - g.double()).square().sum()
        distances = (d.double() - h).square().sum(1)
        nearest = int(distances.argmin())
        margin = float(private_distance - distances[nearest])
        rows.append(dict(coalition=sum(1 << i for i in subset), servers=list(subset),
                         private_distance=float(private_distance),
                         public_distance=float(distances[nearest]), public_slot=nearest,
                         margin=margin, passed=margin >= epsilon))
    aggregate = shares.float().sum(0).double()
    error = aggregate - g.double()
    overlap = (shares != 0).sum(0) > 1
    report = dict(epsilon=epsilon, all_coalitions_pass=all(r['passed'] for r in rows),
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


def solve(g, d, n=3, seed=2026, relative_margin=.01, partitions=8, anchors=16):
    g, d = g.double(), d.double()
    if g.ndim != 1 or d.ndim != 2 or d.shape[1] != len(g) or not len(d):
        raise ValueError('Expected one gradient vector and a nonempty public gradient matrix')
    if not torch.isfinite(g).all() or not torch.isfinite(d).all():
        raise ValueError('Gradients must be finite')
    if not math.isfinite(relative_margin) or relative_margin < 0 or partitions < 1 or anchors < 1:
        raise ValueError('Invalid search budget or relative margin')
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
    best = None
    for _ in range(partitions):
        owners = torch.randint(n, (len(g),), generator=rng)
        base = torch.stack([g * (owners == i) for i in range(n)])
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
            report, rows = audit(h32, g, d, epsilon, h)
            if not report['all_coalitions_pass'] or not report['float32_aggregation_pass']:
                continue
            score = (report['extra_nnz'], max(r['public_distance'] for r in rows),
                     sum(r['public_distance'] for r in rows))
            if best is None or score < best[0]:
                best = (score, dict(shares=h32, exact_shares=h, owners=owners,
                                   repair_coordinate=k, alpha=alpha, magnitude=t,
                                   epsilon=epsilon, scale=scale, audit=report, geometry=rows))
        if best is not None and best[0][0] == 0:
            break
    if best is None:
        raise RuntimeError('No feasible decomposition within search budget; this is not an infeasibility proof')
    return best[1]
