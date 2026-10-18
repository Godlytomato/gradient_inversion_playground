from reconstruction.optimization.missingness.implementations.visible_only import visible_only


def masked_loss(loss_function, reconstructed_gradients, target_gradients, missing_mask=None, **kwargs):
    reconstructed, target = visible_only(
        reconstructed_gradients, target_gradients, missing_mask
    )
    pairs = [(r, t) for r, t in zip(reconstructed, target) if r.numel() > 0]
    if not pairs:
        raise ValueError("No observed gradient coordinates")

    reconstructed, target = zip(*pairs)
    return loss_function(
        reconstructed_gradients=list(reconstructed),
        target_gradients=list(target),
        **kwargs,
    )

