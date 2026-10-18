import torch
import torch.nn.functional as F


def weighted_cosine_loss(
    reconstructed_gradients,
    target_gradients,
    layer_weights,
    state=None,
) -> torch.Tensor:

    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    if len(layer_weights) != len(reconstructed_gradients):
        raise ValueError(
            "Number of layer weights must match number of gradients"
        )

    if len(reconstructed_gradients) == 0:
        raise ValueError(
            "Gradient lists cannot be empty"
        )

    losses = []
    weights = []

    for reconstructed, target, weight in zip(
        reconstructed_gradients,
        target_gradients,
        layer_weights
    ):
        if weight < 0:
            raise ValueError(
                "Layer weights must be nonnegative"
            )

        reconstructed_flat = reconstructed.reshape(-1)
        target_flat = target.reshape(-1)

        similarity = F.cosine_similarity(
            reconstructed_flat,
            target_flat,
            dim=0,
            eps=1e-8,
        )

        losses.append(
            1.0 - similarity
        )

        weights.append(
            float(weight)
        )

    weight_tensor = torch.tensor(
        weights,
        device=losses[0].device,
        dtype=losses[0].dtype,
    )

    if weight_tensor.sum() <= 0:
        raise ValueError(
            "At least one layer weight must be positive"
        )

    loss_tensor = torch.stack(losses)

    return (
        loss_tensor
        * weight_tensor
    ).sum() / weight_tensor.sum()