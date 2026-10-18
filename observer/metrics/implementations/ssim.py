import torch
import torch.nn.functional as F


def ssim(
    reconstruction: torch.Tensor,
    ground_truth: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> float:
    """
    Compute Structural Similarity Index between the reconstruction
    and ground-truth image.

    Higher values indicate better structural similarity.
    """

    if reconstruction.shape != ground_truth.shape:
        raise ValueError(
            "reconstruction and ground_truth must have the same shape"
        )

    if reconstruction.dim() != 4:
        raise ValueError(
            "SSIM expects tensors with shape [batch, channels, height, width]"
        )

    # make sure the window fits the image
    height = reconstruction.shape[-2]
    width = reconstruction.shape[-1]

    size = min(
        window_size,
        height,
        width,
    )

    # SSIM windows should be odd-sized
    if size % 2 == 0:
        size -= 1

    if size < 1:
        raise ValueError(
            "image is too small for SSIM"
        )

    padding = size // 2

    # local means
    mu_x = F.avg_pool2d(
        reconstruction,
        kernel_size=size,
        stride=1,
        padding=padding,
    )

    mu_y = F.avg_pool2d(
        ground_truth,
        kernel_size=size,
        stride=1,
        padding=padding,
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    # local variances
    sigma_x_sq = (
        F.avg_pool2d(
            reconstruction ** 2,
            kernel_size=size,
            stride=1,
            padding=padding,
        )
        - mu_x_sq
    )

    sigma_y_sq = (
        F.avg_pool2d(
            ground_truth ** 2,
            kernel_size=size,
            stride=1,
            padding=padding,
        )
        - mu_y_sq
    )

    sigma_xy = (
        F.avg_pool2d(
            reconstruction * ground_truth,
            kernel_size=size,
            stride=1,
            padding=padding,
        )
        - mu_xy
    )

    C1 = (
        c1
        * data_range ** 2
    )

    C2 = (
        c2
        * data_range ** 2
    )

    numerator = (
        (2 * mu_xy + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1)
        * (sigma_x_sq + sigma_y_sq + C2)
    )

    score = (
        numerator
        / denominator
    )

    return score.mean().item()