import math
import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)

from .transform_utils import (
    batch_tensors,
    jacobian_vector_product,
    reshape_scores,
)


def exact_i2f(
    package: grad_package,
    context: transformation_context,
    samples: int = 10,
    damping: float = 1e-4,
) -> list[torch.Tensor]:
    """
    Approximate I2F influence scores using a randomized
    low-dimensional input subspace.

    The function retains its existing registry name for
    compatibility, but does not construct the full Jacobian.
    """

    if samples <= 0:
        raise ValueError(
            "samples must be greater than zero"
        )
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")

    images, _ = batch_tensors(
        context
    )

    input_size = images.numel()

    samples = min(
        samples,
        input_size,
    )

    basis = torch.randn(
        input_size,
        samples,
        device=images.device,
        dtype=images.dtype,
    )

    basis, _ = torch.linalg.qr(
        basis,
        mode="reduced",
    )

    columns = []

    for index in range(
        samples
    ):

        jv = jacobian_vector_product(
            vector=basis[:, index],
            context=context,
        )

        columns.append(
            jv
        )

    projected = torch.stack(
        columns,
        dim=1,
    )

    gram = (
        projected.T
        @ projected
    )

    identity = torch.eye(
        gram.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    )

    # Solve the regularized system rather than explicitly forming its inverse.
    projected_influence = torch.linalg.solve(
        gram + damping * identity, projected.T,
    ).T

    flat_scores = torch.linalg.vector_norm(
        projected_influence,
        dim=1,
    )

    return reshape_scores(
        flat_scores,
        package,
    )
