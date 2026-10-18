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


def conjugate_gradient(
    matrix_vector_func,
    b: torch.Tensor,
    iterations: int = 10,
    tolerance: float = 1e-6,
) -> torch.Tensor:

    x = torch.zeros_like(
        b
    )

    r = (
        b
        - matrix_vector_func(x)
    )

    p = r.clone()

    rs_old = torch.dot(
        r,
        r,
    )

    for _ in range(iterations):

        Ap = matrix_vector_func(
            p
        )

        denominator = torch.dot(
            p,
            Ap,
        )

        if denominator.abs() < 1e-12:
            break

        alpha = (
            rs_old / denominator
        )

        x = (
            x + alpha * p
        )

        r = (
            r - alpha * Ap
        )

        rs_new = torch.dot(
            r,
            r,
        )

        if torch.sqrt(
            rs_new
        ) < tolerance:
            break

        p = (
            r
            + (
                rs_new / rs_old
            ) * p
        )

        rs_old = rs_new

    return x


def min_svd(
    package: grad_package,
    context: transformation_context,
    iterations: int = 5,
    solve_iterations: int = 8,
    damping: float = 1e-4,
) -> list[torch.Tensor]:
    """
    Approximate the minimum singular direction using
    damped inverse iteration without constructing J.
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

    def damped_jtj(
        vector
    ):
        return (
            jtj_vector_product(
                vector=vector,
                context=context,
            )
            + damping * vector
        )

    for _ in range(iterations):

        z = conjugate_gradient(
            matrix_vector_func=damped_jtj,
            b=v,
            iterations=solve_iterations,
        )

        norm = torch.linalg.vector_norm(
            z
        )

        if norm == 0:
            break

        v = (
            z / norm
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