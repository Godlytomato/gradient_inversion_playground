import torch


def random_initialization(
    batch_size,
    input_shape,
    device
):
    return torch.randn(
        (batch_size, *input_shape),
        device=device,
        requires_grad=True
    )