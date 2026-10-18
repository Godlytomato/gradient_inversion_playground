"""Directional Taylor bound with an explicit raw-gradient residual.

Let eta=a*d, ||d||=1, k=||Dg(x0).T*d||. If Dg is mu-Lipschitz
on the segment to a reconstruction with residual ||g(xhat)-(g0+eta)||=eps,
then max(0,a-eps) <= k*E + mu*E**2/2. No stationary or exact fit is assumed.
Constants/region validity are deliberately NOT certified by this scalar module.
"""
import math


def _nonnegative(**values):
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def residual_aware_bound(noise_l2, residual_l2, directional_gain, mu_j):
    """Return a stable conditional bound, including degenerate cases."""
    _nonnegative(noise_l2=noise_l2, residual_l2=residual_l2,
                 directional_gain=directional_gain, mu_j=mu_j)
    b = max(0.0, noise_l2 - residual_l2)
    if b == 0:
        return {"status": "uninformative", "lower_bound_l2": 0.0}
    if directional_gain == 0 and mu_j == 0:
        return {"status": "incompatible_residual_assumption", "lower_bound_l2": None}
    if mu_j == 0:
        root = b / directional_gain
    else:
        # Compute 2b/(k+hypot(k,sqrt(2*mu*b))) with binary scaling.
        # Neither mu*b nor its square root needs to fit in float64.
        mb, eb = math.frexp(b)
        mm, em = math.frexp(mu_j)
        product_exponent = em + eb
        et = product_exponent // 2
        mt = math.sqrt(2 * mm * mb * (2 if product_exponent % 2 else 1))
        mk, ek = math.frexp(directional_gain)
        exponent = max(ek, et) if directional_gain else et
        k = math.ldexp(mk, ek - exponent) if directional_gain else 0.0
        t = math.ldexp(mt, et - exponent)
        denominator = k + math.hypot(k, t)
        try:
            root = math.ldexp(2 * mb / denominator, eb - exponent)
        except OverflowError:
            root = math.inf
    if not math.isfinite(root) or root == 0:
        return {"status": "outside_numeric_range", "lower_bound_l2": None}
    return {"status": "informative", "lower_bound_l2": root}


def _product(*values):
    if any(value == 0 for value in values):
        return 0.0
    mantissa, exponent = 1.0, 0
    for value in values:
        m, e = math.frexp(value)
        mantissa *= m
        exponent += e
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.inf


def required_noise_l2(target_l2, directional_gain, mu_j,
                      residual_l2=0.0, residual_fraction=0.1):
    """Calibrate only reconstructions satisfying eps <= abs + fraction*a.

    None means the constants cannot give a positive finite error bound;
    infinity means the required amplitude exceeds float64's numeric range.
    """
    _nonnegative(target_l2=target_l2, directional_gain=directional_gain,
                 mu_j=mu_j, residual_l2=residual_l2,
                 residual_fraction=residual_fraction)
    if residual_fraction >= 1:
        raise ValueError("residual_fraction must be less than 1")
    if target_l2 == 0:
        return 0.0
    if directional_gain == 0 and mu_j == 0:
        return None
    terms = (residual_l2, _product(directional_gain, target_l2),
             _product(0.5, mu_j, target_l2, target_l2))
    try:
        value = math.fsum(terms) / (1.0 - residual_fraction)
    except OverflowError:
        value = math.inf
    return math.nextafter(0., math.inf) if value == 0 else value
