"""Resolve the declared minimum-positive singular protocol before run hashing.

Unconfigured schedules adopt the requested minimum-positive protocol in memory.
Explicit strict historical methods remain available for reproducibility.
"""
import math


DEFAULT_MIN_SVD_METHOD = "positive_svd_v1"
CHECKED_MIN_SVD_METHOD = "lanczos_v1"
LEGACY_MIN_SVD_METHOD = "inverse_v2"
DEFAULT_MIN_SVD_RANK_POLICY = "smallest_nonzero"
DEFAULT_RELATIVE_RANK_TOLERANCE = 1e-5
DEFAULT_DENSE_MAX_ELEMENTS = 300_000_000
DEFAULT_DENSE_MAX_INPUT_SIZE = 4096
DEFAULT_KRYLOV_MAX_ITERATIONS = 512
DEFAULT_KRYLOV_CHECK_EVERY = 16
DEFAULT_KRYLOV_MAX_ELEMENTS = 8_000_000
LEGACY_BUDGET_NAMES = ("iterations", "solve_iterations", "damping")
KRYLOV_DEFAULTS = {
    "krylov_max_iterations": DEFAULT_KRYLOV_MAX_ITERATIONS,
    "krylov_check_every": DEFAULT_KRYLOV_CHECK_EVERY,
    "krylov_max_elements": DEFAULT_KRYLOV_MAX_ELEMENTS,
}
DENSE_DEFAULTS = {
    "dense_max_elements": DEFAULT_DENSE_MAX_ELEMENTS,
    "dense_max_input_size": DEFAULT_DENSE_MAX_INPUT_SIZE,
}


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def min_svd_parameters(*, iterations=None, solve_iterations=None, damping=None,
                       solver_method=None, krylov_max_iterations=None,
                       krylov_check_every=None, krylov_max_elements=None,
                       solver_seed=None, rank_policy=None,
                       relative_rank_tolerance=None, dense_max_elements=None,
                       dense_max_input_size=None):
    """Resolve explicit serializable solver, rank convention, and resource caps."""
    legacy_explicit = any(value is not None for value in
                          (iterations, solve_iterations, damping))
    method = ((LEGACY_MIN_SVD_METHOD if legacy_explicit else DEFAULT_MIN_SVD_METHOD)
              if solver_method is None else solver_method)
    if method not in {DEFAULT_MIN_SVD_METHOD, CHECKED_MIN_SVD_METHOD, LEGACY_MIN_SVD_METHOD}:
        raise ValueError(f"Unknown minimum-SVD solver method: {method}")
    policy = ((DEFAULT_MIN_SVD_RANK_POLICY if method == DEFAULT_MIN_SVD_METHOD else "error")
              if rank_policy is None else rank_policy)
    if policy not in {"error", "smallest_nonzero"}:
        raise ValueError("rank_policy must be 'error' or 'smallest_nonzero'")
    if method == DEFAULT_MIN_SVD_METHOD and policy != DEFAULT_MIN_SVD_RANK_POLICY:
        raise ValueError("positive_svd_v1 requires rank_policy='smallest_nonzero'; "
                         "select lanczos_v1 for an explicit full-minimum experiment")
    parameters = {"solver_method": method, "rank_policy": policy}
    if relative_rank_tolerance is not None or policy == "smallest_nonzero":
        cutoff = (DEFAULT_RELATIVE_RANK_TOLERANCE if relative_rank_tolerance is None
                  else relative_rank_tolerance)
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                or not math.isfinite(cutoff) or not 0 < cutoff < 1):
            raise ValueError("relative_rank_tolerance must be finite and between zero and one")
        parameters["relative_rank_tolerance"] = cutoff
    if method == LEGACY_MIN_SVD_METHOD:
        if solver_seed is not None:
            raise ValueError("solver_seed requires a checked minimum-SVD method")
        if any(value is not None for value in
               (krylov_max_iterations, krylov_check_every, krylov_max_elements)):
            raise ValueError("Krylov settings require a checked minimum-SVD method")
        iterations = 5 if iterations is None else iterations
        solve_iterations = 8 if solve_iterations is None else solve_iterations
        damping = 1e-4 if damping is None else damping
        _positive_integer("iterations", iterations)
        _positive_integer("solve_iterations", solve_iterations)
        if (isinstance(damping, bool) or not isinstance(damping, (int, float))
                or not math.isfinite(damping) or damping <= 0):
            raise ValueError("damping must be finite and positive")
        parameters.update(iterations=iterations, solve_iterations=solve_iterations,
                          damping=damping)
    else:
        if legacy_explicit:
            raise ValueError(f"{method} does not use iterations, solve_iterations, or damping; "
                             "remove those settings explicitly or select inverse_v2")
        solver_seed = 0 if solver_seed is None else solver_seed
        if isinstance(solver_seed, bool) or not isinstance(solver_seed, int) or solver_seed < 0:
            raise ValueError("solver_seed must be a nonnegative integer")
        parameters["solver_seed"] = solver_seed
        for name, value in (("krylov_max_iterations", krylov_max_iterations),
                            ("krylov_check_every", krylov_check_every),
                            ("krylov_max_elements", krylov_max_elements)):
            value = KRYLOV_DEFAULTS[name] if value is None else value
            _positive_integer(name, value)
            parameters[name] = value
    if method == DEFAULT_MIN_SVD_METHOD:
        for name, value in (("dense_max_elements", dense_max_elements),
                            ("dense_max_input_size", dense_max_input_size)):
            value = DENSE_DEFAULTS[name] if value is None else value
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer (zero disables dense solving)")
            parameters[name] = value
    elif dense_max_elements is not None or dense_max_input_size is not None:
        raise ValueError("Dense positive-SVD settings require solver_method='positive_svd_v1'")
    return parameters


def normalize_min_svd_parameters(parameters):
    """Adopt the authorized positive protocol unless strict rank policy is explicit."""
    method = parameters.get("solver_method")
    if method is not None and method not in {
            DEFAULT_MIN_SVD_METHOD, CHECKED_MIN_SVD_METHOD, LEGACY_MIN_SVD_METHOD}:
        raise ValueError(f"Unknown minimum-SVD solver method: {method}")
    if parameters.get("rank_policy") is None:
        for name in LEGACY_BUDGET_NAMES:
            parameters.pop(name, None)
        parameters.update(solver_method=DEFAULT_MIN_SVD_METHOD,
                          rank_policy=DEFAULT_MIN_SVD_RANK_POLICY)
    elif parameters.get("solver_method") is None:
        parameters["solver_method"] = (DEFAULT_MIN_SVD_METHOD
                                       if parameters["rank_policy"] == "smallest_nonzero"
                                       else LEGACY_MIN_SVD_METHOD)
    names = (*LEGACY_BUDGET_NAMES, *KRYLOV_DEFAULTS, *DENSE_DEFAULTS,
             "solver_seed", "solver_method", "rank_policy", "relative_rank_tolerance")
    resolved = min_svd_parameters(**{name: parameters[name]
                                     for name in names if name in parameters})
    parameters.update(resolved)
    return parameters


def normalize_min_svd_settings(description):
    """Version minimum ranking/construction parameters before validation and hashing."""
    for transformation in description.gradient_transformations:
        ranking = transformation.ranking_policy
        if ranking.name == "min_svd" and isinstance(ranking.parameters, dict):
            normalize_min_svd_parameters(ranking.parameters)
        transform = transformation.transform
        if transform.name != "noise":
            continue
        construction = transform.parameters.get("construction")
        if not isinstance(construction, dict) or construction.get("name") != "min_svd_alignment":
            continue
        parameters = construction.setdefault("parameters", {})
        if isinstance(parameters, dict):
            normalize_min_svd_parameters(parameters)
    return description
