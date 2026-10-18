import math
import torch


def psnr(
    reconstruction: torch.Tensor,
    ground_truth: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between the reconstruction
    and ground-truth image.

    Higher values indicate better reconstruction quality.
    """

    mse = torch.mean(
        (reconstruction - ground_truth) ** 2
    ).item()

    if mse == 0:
        return float("inf")

    return (
        10.0
        * math.log10(
            (data_range ** 2)
            / mse
        )
    )