from reconstruction.optimization.missingness.implementations.visible_only import visible_only


def masked_loss(loss_function, reconstructed_gradients, target_gradients, missing_mask=None, **kwargs):
    reconstructed, target = visible_only(
        reconstructed_gradients, target_gradients, missing_mask
    )
    retained = [i for i, r in enumerate(reconstructed) if r.numel() > 0]
    pairs = [(reconstructed[i], target[i]) for i in retained]
    if not pairs:
        raise ValueError("No observed gradient coordinates")

    # Layer metadata refers to the original parameter tensors. Keep it aligned
    # when an entire tensor is hidden by the observation mask.
    if "layer_weights" in kwargs:
        weights = kwargs["layer_weights"]
        if len(weights) != len(reconstructed):
            raise ValueError("Number of layer weights must match number of gradients")
        kwargs["layer_weights"] = [weights[i] for i in retained]

    reconstructed, target = zip(*pairs)
    return loss_function(
        reconstructed_gradients=list(reconstructed),
        target_gradients=list(target),
        **kwargs,
    )

