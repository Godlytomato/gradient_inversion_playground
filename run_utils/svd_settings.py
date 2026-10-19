"""Version minimum-SVD solvers before experiment validation and hashing.

New presets use the checked Krylov solver. Existing schedules retain their
inverse-iteration method until the user explicitly migrates their settings.
"""
import math


DEFAULT_MIN_SVD_METHOD = "lanczos_v1"
LEGACY_MIN_SVD_METHOD = "inverse_v2"
DEFAULT_KRYLOV_MAX_ITERATIONS = 512
DEFAULT_KRYLOV_CHECK_EVERY = 16
DEFAULT_KRYLOV_MAX_ELEMENTS = 8_000_000
LEGACY_BUDGET_NAMES = ("iterations", "solve_iterations", "damping")
KRYLOV_DEFAULTS = {
    "krylov_max_iterations": DEFAULT_KRYLOV_MAX_ITERATIONS,
    "krylov_check_every": DEFAULT_KRYLOV_CHECK_EVERY,
    "krylov_max_elements": DEFAULT_KRYLOV_MAX_ELEMENTS,
}


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def min_svd_parameters(*, iterations=None, solve_iterations=None, damping=None,
                       solver_method=None, krylov_max_iterations=None,
                       krylov_check_every=None, krylov_max_elements=None,
                       solver_seed=None):
    """Resolve a serializable method without discarding explicit old budgets."""
    legacy_explicit = any(value is not None for value in
                          (iterations, solve_iterations, damping))
    method = ((LEGACY_MIN_SVD_METHOD if legacy_explicit else DEFAULT_MIN_SVD_METHOD)
              if solver_method is None else solver_method)
    if method not in {DEFAULT_MIN_SVD_METHOD, LEGACY_MIN_SVD_METHOD}:
        raise ValueError(f"Unknown minimum-SVD solver method: {method}")
    parameters = {"solver_method": method}
    if method == LEGACY_MIN_SVD_METHOD:
        if solver_seed is not None:
            raise ValueError("solver_seed requires solver_method='lanczos_v1'")
        if any(value is not None for value in
               (krylov_max_iterations, krylov_check_every, krylov_max_elements)):
            raise ValueError("Krylov settings require solver_method='lanczos_v1'")
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
            raise ValueError("lanczos_v1 does not use iterations, solve_iterations, or damping; "
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
    return parameters


def normalize_min_svd_parameters(parameters):
    """Resolve defaults in place while leaving unknown arguments to validation."""
    method = parameters.setdefault("solver_method", LEGACY_MIN_SVD_METHOD)
    names = (*LEGACY_BUDGET_NAMES, *KRYLOV_DEFAULTS, "solver_seed", "solver_method")
    resolved = min_svd_parameters(**{name: parameters[name]
                                     for name in names if name in parameters})
    parameters.update(resolved)
    return parameters


def normalize_min_svd_settings(description):
    """Preserve old YAML methods and version explicit new methods before hashing."""
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
