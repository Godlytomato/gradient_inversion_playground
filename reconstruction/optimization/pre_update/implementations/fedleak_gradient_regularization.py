import torch


def fedleak_gradient_regularization(
    reconstruction,
    context,
    state=None,
    perturbation_size=0.01,
    blend=0.5
):

    if reconstruction.grad is None:
        return

    # gradient of D(x')
    original_gradient = (
        reconstruction.grad
        .detach()
        .clone()
    )

    norm = torch.norm(
        original_gradient,
        p=2
    )

    if norm == 0:
        return

    # phi = k * grad / ||grad||
    perturbation = (
        perturbation_size
        * original_gradient
        / norm
    )

    # x' + phi
    perturbed_reconstruction = (
        reconstruction.detach()
        + perturbation
    )

    perturbed_reconstruction.requires_grad_(
        True
    )

    # D(x' + phi)
    perturbed_loss = context.objective(
        perturbed_reconstruction
    )

    # gradient of D(x' + phi)
    perturbed_gradient = torch.autograd.grad(
        perturbed_loss,
        perturbed_reconstruction
    )[0]

    # blend the two gradient directions
    regularized_gradient = (
        (1.0 - blend)
        * original_gradient
        + blend
        * perturbed_gradient
    )

    # replace the gradient used by the update
    with torch.no_grad():
        reconstruction.grad.copy_(
            regularized_gradient
        )