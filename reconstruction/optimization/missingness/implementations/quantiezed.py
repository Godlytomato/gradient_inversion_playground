import torch

from reconstruction.optimization.missingness.implementations.visible_only import (
    visible_only,
)


def quantized(
    reconstructed_gradients,
    target_gradients,
    missing_mask,
    mode="bits",
    bits=8,
    step=None,
    stochastic=False,
):
    """
    Match reconstructed gradients through the same quantized
    observation model as the target gradients.

    Uses a straight-through estimator so rounding does not
    destroy the reconstruction gradient.
    """

    if stochastic:
        raise NotImplementedError(
            "Stochastic quantization-aware matching "
            "is not implemented yet."
        )

    observed = []

    for reconstructed, target in zip(
        reconstructed_gradients,
        target_gradients,
    ):

        if mode == "step":

            if step is None or step <= 0:
                raise ValueError(
                    "step quantization requires step > 0"
                )

            scale = step

        elif mode == "bits":

            if bits < 2:
                raise ValueError(
                    "bit quantization requires bits >= 2"
                )

            levels = (
                2 ** (bits - 1)
            ) - 1

            target_max = (
                target
                .detach()
                .abs()
                .max()
            )

            if target_max == 0:
                observed.append(
                    reconstructed
                )
                continue

            scale = (
                target_max
                / levels
            )

        else:
            raise ValueError(
                f"Unknown quantization mode: {mode}"
            )

        scaled = (
            reconstructed
            / scale
        )

        quantized = (
            torch.round(
                scaled
            )
            * scale
        )

        if mode == "bits":

            quantized = torch.clamp(
                quantized / scale,
                min=-levels,
                max=levels,
            ) * scale

        # forward:
        #     uses quantized value
        #
        # backward:
        #     acts like identity
        observed_gradient = (
            reconstructed
            + (
                quantized
                - reconstructed
            ).detach()
        )

        observed.append(
            observed_gradient
        )

    return visible_only(
        reconstructed_gradients=observed,
        target_gradients=target_gradients,
        missing_mask=missing_mask,
    )