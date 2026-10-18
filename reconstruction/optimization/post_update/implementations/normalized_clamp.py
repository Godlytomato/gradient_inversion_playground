import torch


def normalized_clamp(
    reconstruction,
    normalization,
    state=None
):
    mean = torch.as_tensor(
        normalization["mean"],
        device=reconstruction.device,
        dtype=reconstruction.dtype
    ).view(1, -1, 1, 1)

    std = torch.as_tensor(
        normalization["std"],
        device=reconstruction.device,
        dtype=reconstruction.dtype
    ).view(1, -1, 1, 1)

    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    with torch.no_grad():
        reconstruction.copy_(
            torch.maximum(
                torch.minimum(
                    reconstruction,
                    upper
                ),
                lower
            )
        )