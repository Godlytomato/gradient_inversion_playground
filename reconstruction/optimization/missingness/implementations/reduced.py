from reconstruction.optimization.missingness.implementations.visible_only import (
    visible_only,
)


def reduced(
    reconstructed_gradients,
    target_gradients,
    missing_mask,
    factor=0.5,
):
    """
    Apply the known gradient reduction factor to reconstructed
    gradients before matching against the observed target.
    """

    if factor < 0 or factor > 1:
        raise ValueError(
            "Reduction factor must be between 0 and 1"
        )

    reconstructed_gradients = [
        gradient * factor
        for gradient
        in reconstructed_gradients
    ]

    return visible_only(
        reconstructed_gradients=(
            reconstructed_gradients
        ),
        target_gradients=target_gradients,
        missing_mask=missing_mask,
    )