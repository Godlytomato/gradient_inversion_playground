import torch
import torch.nn.functional as F


def layerwise_cosine_loss(
    reconstructed_gradients,
    target_gradients,
    state=None,
) -> torch.Tensor:

    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    if len(reconstructed_gradients) == 0:
        raise ValueError(
            "Gradient lists cannot be empty"
        )

    losses = []

    for reconstructed, target in zip(
        reconstructed_gradients,
        target_gradients
    ):
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

    return torch.stack(losses).mean()