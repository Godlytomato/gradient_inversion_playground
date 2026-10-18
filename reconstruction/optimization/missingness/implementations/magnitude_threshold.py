def magnitude_threshold(
    reconstructed_gradients,
    target_gradients,
    missing_mask,
    threshold=0.01,
):
    """
    Ignore observed gradient coordinates whose absolute
    magnitude is below the given threshold.

    Existing missing coordinates remain missing.
    """

    if threshold < 0:
        raise ValueError(
            "Magnitude threshold must be non-negative"
        )

    filtered_reconstructed = []
    filtered_target = []

    for index, (
        reconstructed,
        target,
    ) in enumerate(
        zip(
            reconstructed_gradients,
            target_gradients,
        )
    ):

        # values large enough to be considered visible
        visible = (
            target.abs()
            >= threshold
        )

        # also respect any existing pruning mask
        if missing_mask is not None:

            visible = (
                visible
                & ~missing_mask[index]
            )

        filtered_reconstructed.append(
            reconstructed[
                visible
            ]
        )

        filtered_target.append(
            target[
                visible
            ]
        )

    return (
        filtered_reconstructed,
        filtered_target,
    )