import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def quantization_delta(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    mode: str = "bits",
    bits: int = 8,
    step: float | None = None,
    stochastic: bool = False,
    **kwargs
) -> list[torch.Tensor]:

    deltas = []

    for gradient in package.grads:

        # -----------------------------------------
        # explicit step-size quantization
        # -----------------------------------------
        if mode == "step":

            if step is None or step <= 0:
                raise ValueError(
                    "step mode requires step > 0"
                )

            scaled = gradient / step

            if stochastic:
                lower = torch.floor(scaled)
                probability = scaled - lower

                random_values = torch.rand_like(probability)

                quantized_scaled = (
                    lower
                    + (random_values < probability).to(
                        gradient.dtype
                    )
                )

            else:
                quantized_scaled = torch.round(scaled)

            quantized = quantized_scaled * step

        # -----------------------------------------
        # bit-width quantization
        # -----------------------------------------
        elif mode == "bits":

            if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2:
                raise ValueError(
                    "Symmetric quantization requires an integer bit width >= 2"
                )

            max_abs = torch.max(torch.abs(gradient))

            # no change needed for all-zero gradients
            if max_abs == 0:
                deltas.append(
                    torch.zeros_like(gradient)
                )
                continue

            # signed symmetric quantization
            levels = (2 ** (bits - 1)) - 1

            if levels <= 0:
                raise ValueError(
                    "bit quantization requires bits >= 2"
                )

            scale = max_abs / levels

            scaled = gradient / scale

            if stochastic:
                lower = torch.floor(scaled)
                probability = scaled - lower

                random_values = torch.rand_like(probability)

                quantized_scaled = (
                    lower
                    + (random_values < probability).to(
                        gradient.dtype
                    )
                )

            else:
                quantized_scaled = torch.round(scaled)

            quantized_scaled = torch.clamp(
                quantized_scaled,
                min=-levels,
                max=levels
            )

            quantized = quantized_scaled * scale

        else:
            raise ValueError(
                f"Unknown quantization mode: {mode}"
            )

        # delta such that:
        #
        # gradient + delta = quantized gradient
        delta = quantized - gradient

        deltas.append(delta)

    return deltas
