import torch
import torch.nn.functional as F


def cosine_loss(
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

    reconstructed = torch.cat([
        gradient.reshape(-1)
        for gradient in reconstructed_gradients
    ])

    target = torch.cat([
        gradient.reshape(-1)
        for gradient in target_gradients
    ])

    similarity = F.cosine_similarity(
        reconstructed,
        target,
        dim=0,
        eps=1e-8,
    )

    return 1.0 - similarity