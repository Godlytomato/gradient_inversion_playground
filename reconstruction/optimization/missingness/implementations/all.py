
import torch

from reconstruction.optimization.missingness.implementations.visible_only import (
    visible_only,
)
from reconstruction.optimization.missingness.implementations.clipped import (
    clipped,
)
from reconstruction.optimization.missingness.implementations.quantiezed import (
    quantized,
)
from reconstruction.optimization.missingness.implementations.reduced import (
    reduced,
)


def all_policies(
    reconstructed_gradients,
    target_gradients,
    missing_mask=None,
    threshold: float = 0.0,
    clipping: dict | None = None,
    quantization: dict | None = None,
    reduction: dict | None = None,
    order: tuple = ("clipping", "quantization", "reduction"),
):
    """
    Apply known defense observation models, then filter
    missing and zero-valued observed gradient coordinates.
    """

    if threshold < 0:
        raise ValueError("Threshold must be nonnegative")

    # Validate gradients and missing masks.
    visible_only(
        reconstructed_gradients,
        target_gradients,
        missing_mask,
    )

    reconstructed = list(reconstructed_gradients)
    target = list(target_gradients)

    policies = {
        "clipping": (clipped, clipping),
        "quantization": (quantized, quantization),
        "reduction": (reduced, reduction),
    }

    if len(order) != len(set(order)):
        raise ValueError("Duplicate policies in order")

    for name in order:
        if name not in policies:
            raise ValueError(f"Unknown policy: {name}")

    # Apply configured observation models in order.
    for name in order:
        function, parameters = policies[name]

        if parameters is None:
            continue

        reconstructed, target = function(
            reconstructed_gradients=reconstructed,
            target_gradients=target,
            missing_mask=None,
            **parameters,
        )

    # Apply visible_only and ignore zeros.
    filtered_reconstructed = []
    filtered_target = []

    for i, (r, t) in enumerate(zip(reconstructed, target)):

        visible = t.abs() > threshold

        if missing_mask is not None:
            visible = visible & ~missing_mask[i]

        filtered_reconstructed.append(r[visible])
        filtered_target.append(t[visible])

    if not any(r.numel() > 0 for r in filtered_reconstructed):
        raise ValueError("No observed gradient coordinates")

    return filtered_reconstructed, filtered_target
