import torch
import torch.nn.functional as F

from reconstruction.optimization.initialization.initialization_def import parameterized_image


def uniform_initialization(batch_size, input_shape, device):
    """Raw pixels away from the sigmoid's saturated boundaries."""
    return .05 + .9 * torch.rand((batch_size, *input_shape), device=device)


def uniform_sigmoid_initialization(batch_size, input_shape, device):
    image = uniform_initialization(batch_size, input_shape, device)
    return parameterized_image(torch.logit(image).detach().requires_grad_(True), torch.sigmoid)


def smooth_uniform_initialization(batch_size, input_shape, device):
    """Seeded low-frequency pixels, without private/public image information.

    Only initialization is smoothed. The optimizer still controls every pixel,
    so fine spatial structure can be recovered wherever the gradient supports it.
    """
    channels, height, width = input_shape
    grid = .05 + .9 * torch.rand(
        (batch_size, channels, min(4, height), min(4, width)), device=device,
    )
    return F.interpolate(grid, size=(height, width), mode='bilinear', align_corners=False)


def smooth_uniform_sigmoid_initialization(batch_size, input_shape, device):
    image = smooth_uniform_initialization(batch_size, input_shape, device)
    return parameterized_image(torch.logit(image).detach().requires_grad_(True), torch.sigmoid)
