"""Explicit, versioned settings for conditional lower-bound calibration.

Defaults affect the scientific meaning of a run, so they must be part of its
stored description and identity before completed experiments are deduplicated.
Historical database rows are deliberately left unchanged.
"""
import math


DEFAULT_BOUND_METHOD = "directional_residual_v1"
LEGACY_BOUND_METHOD = "legacy_proof_v1"
BOUND_SCALERS = {"lower_bound_threshold", "max_svd_reference_scale"}


def bound_parameters(*, target_error, mu_j, bound_method=DEFAULT_BOUND_METHOD,
                     residual_l2=0.0, residual_fraction=0.1):
    """Build a serializable manual-constant calibration configuration."""
    if bound_method not in {DEFAULT_BOUND_METHOD, LEGACY_BOUND_METHOD}:
        raise ValueError(f"Unknown lower-bound method: {bound_method}")
    for name, value in (("target_error", target_error), ("mu_j", mu_j),
                        ("residual_l2", residual_l2),
                        ("residual_fraction", residual_fraction)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if residual_fraction >= 1:
        raise ValueError("residual_fraction must be less than 1")
    parameters = {
        "target_error": target_error,
        "mu_j_method": "manual",
        "mu_j_debug": False,
        "mu_j": mu_j,
        "bound_method": bound_method,
    }
    if bound_method == DEFAULT_BOUND_METHOD:
        parameters.update(residual_l2=residual_l2, residual_fraction=residual_fraction)
    return parameters


def normalize_bound_settings(description):
    """Resolve implicit bound defaults in place before validation and hashing.

    This upgrades existing YAML without rewriting schedules. Only descriptions
    containing bound scalers change identity; unrelated experiments remain
    eligible for their existing completed-result skip. Explicit legacy choices
    remain available, without adding unused residual parameters to their hash.
    """
    for transformation in description.gradient_transformations:
        transform = transformation.transform
        if transform.name != "noise":
            continue
        scaling = transform.parameters.get("scaling")
        if not isinstance(scaling, dict) or scaling.get("name") not in BOUND_SCALERS:
            continue
        parameters = scaling.setdefault("parameters", {})
        if not isinstance(parameters, dict):
            # Let ordinary description validation explain malformed settings.
            continue
        method = parameters.setdefault("bound_method", DEFAULT_BOUND_METHOD)
        if method == DEFAULT_BOUND_METHOD:
            parameters.setdefault("residual_l2", 0.0)
            parameters.setdefault("residual_fraction", 0.1)
    return description
