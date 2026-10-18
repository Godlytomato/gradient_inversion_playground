import torch


def exponential_scheduler(
    optimizer,
    gamma=0.99
):
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )