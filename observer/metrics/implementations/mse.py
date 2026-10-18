import torch


def mse(reconstruction: torch.Tensor,
        ground_truth: torch.Tensor) -> float:

    return torch.mean(
        (reconstruction - ground_truth) ** 2
    ).item()