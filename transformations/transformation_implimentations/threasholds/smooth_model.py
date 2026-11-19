"""Create a smooth, weight-identical copy for analytic bounds only."""
import copy
import math

import torch
from torch import nn


class SmoothSoftplus(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        # Stable, smoothly differentiable (no hard threshold branch).
        return torch.logsumexp(torch.stack((torch.zeros_like(x), self.beta * x)), dim=0) / self.beta


def softplus_copy(model: nn.Module, beta: float = 10.0) -> nn.Module:
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("softplus_beta must be positive and finite")

    surrogate = copy.deepcopy(model)

    def replace_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, SmoothSoftplus(beta))
            else:
                replace_relu(child)

    replace_relu(surrogate)
    # Fix running BatchNorm statistics during repeated Jacobian evaluations.
    surrogate.eval()
    return surrogate
