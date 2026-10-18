import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from .transform_utils import (
    batch_tensors,
    jacobian_vector_product,
    jtj_vector_product,
    reshape_scores,
)


def max_svd(
    package: grad_package,
    context: transformation_context,
    iterations: int = 10,
) -> list[torch.Tensor]:
    """
    Estimate contribution to the largest singular direction
    without explicitly constructing the Jacobian.
    """

    images, _ = batch_tensors(
        context
    )

    size = images.numel()

    v = torch.randn(
        size,
        device=images.device,
        dtype=images.dtype,
    )

    v = (
        v
        / torch.linalg.vector_norm(v)
    )

    for _ in range(iterations):

        v_new = jtj_vector_product(
            vector=v,
            context=context,
        )

        norm = torch.linalg.vector_norm(
            v_new
        )

        if norm == 0:
            break

        v = (
            v_new / norm
        )

    u = jacobian_vector_product(
        vector=v,
        context=context,
    )

    norm = torch.linalg.vector_norm(
        u
    )

    if norm != 0:
        u = u / norm

    flat_scores = torch.abs(
        u
    )

    return reshape_scores(
        flat_scores,
        package,
    )