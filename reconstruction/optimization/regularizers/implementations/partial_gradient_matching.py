import torch


def partial_gradient_matching(
    reconstructed_gradients,
    target_gradients,
    state=None,
    percent=0.1
):
    if len(reconstructed_gradients) != len(target_gradients):
        raise ValueError(
            "Reconstructed and target gradients must have the same length"
        )

    if percent <= 0 or percent > 1:
        raise ValueError(
            "percent must be greater than 0 and at most 1"
        )

    reconstructed_flat = torch.cat([
        gradient.reshape(-1)
        for gradient in reconstructed_gradients
    ])

    target_flat = torch.cat([
        gradient.reshape(-1)
        for gradient in target_gradients
    ])

    count = max(
        1,
        int(target_flat.numel() * percent)
    )

    indices = torch.topk(
        torch.abs(target_flat),
        k=count,
        largest=True
    ).indices

    reconstructed_selected = reconstructed_flat[
        indices
    ]

    target_selected = target_flat[
        indices
    ]

    return torch.sum(
        (
            reconstructed_selected
            - target_selected
        ) ** 2
    )