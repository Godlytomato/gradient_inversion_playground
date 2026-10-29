import torch

from reconstruction.optimization.initialization.initialization_def import parameterized_image


def uniform_initialization(batch_size, input_shape, device):
    """Raw pixels away from the sigmoid's saturated boundaries."""
    return .05 + .9 * torch.rand((batch_size, *input_shape), device=device)


def uniform_sigmoid_initialization(batch_size, input_shape, device):
    image = uniform_initialization(batch_size, input_shape, device)
    return parameterized_image(torch.logit(image).detach().requires_grad_(True), torch.sigmoid)
