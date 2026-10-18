import torch

from transformations.transform_def import (
    grad_package,
    transformation_context
)


def relative_gradient_scale(
    package: grad_package,
    noise_tensors: list[torch.Tensor],
    selection_masks: list[torch.Tensor],
    context: transformation_context,
    relative_scale: float,
    mode: str = "layer",
    **kwargs
) -> list[torch.Tensor]:

    if relative_scale < 0:
        raise ValueError(
            "relative_scale must be non-negative"
        )

    scaled_noise = []

    if mode == "layer":

        for gradient, noise in zip(
            package.grads,
            noise_tensors
        ):

            grad_norm = torch.norm(gradient)
            noise_norm = torch.norm(noise)

            if noise_norm == 0:
                scaled_noise.append(
                    torch.zeros_like(noise)
                )
                continue

            target_norm = relative_scale * grad_norm

            scaled_noise.append(
                noise * (target_norm / noise_norm)
            )

        return scaled_noise

    elif mode == "global":

        grad_norm = torch.sqrt(
            sum(
                torch.sum(g ** 2)
                for g in package.grads
            )
        )

        noise_norm = torch.sqrt(
            sum(
                torch.sum(n ** 2)
                for n in noise_tensors
            )
        )

        if noise_norm == 0:
            return [
                torch.zeros_like(noise)
                for noise in noise_tensors
            ]

        target_norm = relative_scale * grad_norm
        scale = target_norm / noise_norm

        return [
            noise * scale
            for noise in noise_tensors
        ]

    else:
        raise ValueError(
            f"Unknown relative scaling mode: {mode}"
        )