import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from transformations.ranking_implimentations.transform_utils import (
    min_singular_direction,
)

from transformations.transformation_implimentations.tranform_utils import (
    selected_indices,
    vector_to_noise,
)


def min_svd_alignment(
    package: grad_package,
    selection_masks: list[torch.Tensor],
    context: transformation_context,
) -> list[torch.Tensor]:

    indices = selected_indices(
        package,
        selection_masks,
    )

    if indices.numel() == 0:
        return [
            torch.zeros_like(gradient)
            for gradient in package.grads
        ]

    # u is the singular direction in gradient space
    _, u, _ = min_singular_direction(
        context
    )

    direction = u[
        indices
    ]

    norm = torch.linalg.vector_norm(
        direction
    )

    if norm != 0:
        direction = direction / norm

    return vector_to_noise(
        package,
        selection_masks,
        direction,
    )