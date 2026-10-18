import torch


def clamp_update(
    reconstruction: torch.Tensor,
    minimum: float = 0.0,
    maximum: float = 1.0,
    state=None
) -> torch.Tensor:

    with torch.no_grad():
        reconstruction.clamp_(
            minimum,
            maximum
        )

    return reconstruction