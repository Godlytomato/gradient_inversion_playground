import torch


def adam_optimizer(
    parameters,
    lr: float = 0.1
):
    return torch.optim.Adam(
        parameters,
        lr=lr
    )