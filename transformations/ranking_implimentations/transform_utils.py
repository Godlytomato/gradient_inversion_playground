import torch
import torch.nn as nn

from transformations.transform_def import (
    grad_package,
    transformation_context,
)


# ============================================================
# basic helpers
# ============================================================

def flatten_gradients(
    package: grad_package
) -> torch.Tensor:
    """
    Flatten all parameter gradients into one vector.
    """

    return torch.cat([
        grad.reshape(-1)
        for grad in package.grads
    ])


def reshape_scores(
    flat_scores: torch.Tensor,
    package: grad_package
) -> list[torch.Tensor]:
    """
    Reshape a flat score vector back to gradient tensor shapes.
    """

    scores = []
    position = 0

    for grad in package.grads:

        size = grad.numel()

        score = flat_scores[
            position:position + size
        ].reshape(
            grad.shape
        )

        scores.append(
            score
        )

        position += size

    return scores


def gradient_size(
    package: grad_package
) -> int:
    """
    Return the total number of gradient values.
    """

    return sum(
        grad.numel()
        for grad in package.grads
    )


def batch_tensors(
    context: transformation_context
):
    """
    Convert context.batch into image and label tensors.
    """

    model = context.model

    device = next(
        model.parameters()
    ).device

    images = []
    labels = []

    for image, label in context.batch:

        images.append(
            image
        )

        labels.append(
            int(label)
        )

    if not images:
        raise ValueError(
            "Transformation batch cannot be empty"
        )

    images = torch.stack(
        images
    ).to(
        device
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
        device=device,
    )

    return images, labels


def input_size(
    context: transformation_context
) -> int:
    """
    Return the total number of input values in the batch.
    """

    images, _ = batch_tensors(
        context
    )

    return images.numel()


# ============================================================
# loss / gradient helper
# ============================================================

def model_gradient_vector(
    model,
    images,
    labels,
    loss_function=None,
    create_graph=True,
):
    """
    Return the model parameter gradients as one flat vector.
    """

    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    model.zero_grad(
        set_to_none=True
    )

    predictions = model(
        images
    )

    loss = loss_function(
        predictions,
        labels,
    )

    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        create_graph=create_graph,
    )

    return torch.cat([
        gradient.reshape(-1)
        for gradient in gradients
    ])


# ============================================================
# exact Jacobian
#
# only use this for small validation problems
# ============================================================

def gradient_jacobian(
    context: transformation_context
) -> torch.Tensor:
    """
    Construct the full Jacobian of model gradients
    with respect to the input.

    Shape:
        [gradient values, input values]

    This is extremely expensive and should only be
    used for small validation cases.
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    images = (
        images
        .detach()
        .clone()
    )

    images.requires_grad_(
        True
    )

    flat_gradients = model_gradient_vector(
        model=model,
        images=images,
        labels=labels,
        loss_function=context.loss_function,
        create_graph=True,
    )

    rows = []

    for gradient_value in flat_gradients:

        input_gradient = torch.autograd.grad(
            gradient_value,
            images,
            retain_graph=True,
        )[0]

        rows.append(
            input_gradient.reshape(-1)
        )

    return torch.stack(
        rows
    )


def exact_jtj_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J^T J)v using the exact Jacobian.

    Validation only.
    """

    jacobian = gradient_jacobian(
        context
    )

    vector = vector.to(
        jacobian.device
    )

    return (
        jacobian.T
        @ (
            jacobian
            @ vector
        )
    )


def exact_jjt_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J J^T)v using the exact Jacobian.

    Validation only.
    """

    jacobian = gradient_jacobian(
        context
    )

    vector = vector.to(
        jacobian.device
    )

    return (
        jacobian
        @ (
            jacobian.T
            @ vector
        )
    )


# ============================================================
# matrix-free Jacobian products
# ============================================================

def jacobian_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute Jv without explicitly constructing J.

    Input:
        vector in input space

    Output:
        vector in gradient space
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    vector = (
        vector
        .to(
            device=images.device,
            dtype=images.dtype,
        )
        .reshape_as(
            images
        )
    )

    def gradient_function(
        input_images
    ):

        return model_gradient_vector(
            model=model,
            images=input_images,
            labels=labels,
            loss_function=context.loss_function,
            create_graph=True,
        )

    _, jv = torch.autograd.functional.jvp(
        gradient_function,
        images,
        vector,
        create_graph=False,
        strict=False,
    )

    return jv


def jacobian_transpose_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute J^T v without explicitly constructing J.

    Input:
        vector in gradient space

    Output:
        flattened vector in input space
    """

    model = context.model

    images, labels = batch_tensors(
        context
    )

    images = (
        images
        .detach()
        .clone()
    )

    images.requires_grad_(
        True
    )

    flat_gradients = model_gradient_vector(
        model=model,
        images=images,
        labels=labels,
        loss_function=context.loss_function,
        create_graph=True,
    )

    vector = (
        vector
        .to(
            device=flat_gradients.device,
            dtype=flat_gradients.dtype,
        )
        .reshape_as(
            flat_gradients
        )
    )

    jtv = torch.autograd.grad(
        outputs=flat_gradients,
        inputs=images,
        grad_outputs=vector,
        create_graph=False,
    )[0]

    return jtv.reshape(-1)


def jtj_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J^T J)v without constructing J.
    """

    jv = jacobian_vector_product(
        vector=vector,
        context=context,
    )

    return jacobian_transpose_vector_product(
        vector=jv,
        context=context,
    )


def jjt_vector_product(
    vector: torch.Tensor,
    context: transformation_context,
) -> torch.Tensor:
    """
    Compute (J J^T)v without constructing J.
    """

    jtv = jacobian_transpose_vector_product(
        vector=vector,
        context=context,
    )

    return jacobian_vector_product(
        vector=jtv,
        context=context,
    )


# ============================================================
# estimated Jacobian magnitude
# ============================================================

def estimated_jacobian_magnitude(
    context: transformation_context,
    samples: int = 10,
) -> torch.Tensor:
    """
    Estimate the L2 norm of every Jacobian row using
    randomized Rademacher probes.

    Returns one score for every gradient coordinate.
    """

    if samples <= 0:
        raise ValueError(
            "samples must be greater than zero"
        )

    images, _ = batch_tensors(
        context
    )

    estimate = None

    for _ in range(samples):

        vector = torch.empty_like(
            images
        )

        vector.bernoulli_(
            0.5
        )

        vector = (
            vector * 2
            - 1
        )

        jv = jacobian_vector_product(
            vector=vector,
            context=context,
        )

        squared = (
            jv ** 2
        )

        if estimate is None:
            estimate = squared
        else:
            estimate = (
                estimate
                + squared
            )

    estimate = (
        estimate
        / samples
    )

    return torch.sqrt(
        estimate
    )


# ============================================================
# conjugate gradient
# ============================================================

def conjugate_gradient(
    matrix_vector_func,
    b: torch.Tensor,
    iterations: int = 8,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """
    Approximately solve Ax=b using only a function
    that computes Ax.
    """

    x = torch.zeros_like(
        b
    )

    r = (
        b
        - matrix_vector_func(
            x
        )
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
            rs_old
            / denominator
        )

        x = (
            x
            + alpha * p
        )

        r = (
            r
            - alpha * Ap
        )

        rs_new = torch.dot(
            r,
            r,
        )

        if torch.sqrt(
            rs_new
        ) < tolerance:
            break

        if rs_old.abs() < 1e-12:
            break

        p = (
            r
            + (
                rs_new
                / rs_old
            ) * p
        )

        rs_old = rs_new

    return x


# ============================================================
# singular directions
# ============================================================

def max_singular_direction(
    context: transformation_context,
    iterations: int = 10,
):
    """
    Estimate the dominant singular triplet of J.

    Returns:
        sigma:
            estimated largest singular value

        u:
            gradient-space left singular direction

        v:
            input-space right singular direction
    """

    images, _ = batch_tensors(
        context
    )

    v = torch.randn(
        images.numel(),
        device=images.device,
        dtype=images.dtype,
    )

    norm = torch.linalg.vector_norm(
        v
    )

    if norm == 0:
        raise RuntimeError(
            "Could not initialize max singular direction"
        )

    v = (
        v / norm
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

    sigma = torch.linalg.vector_norm(
        u
    )

    if sigma != 0:
        u = (
            u / sigma
        )

    return sigma, u, v


def min_singular_direction(
    context: transformation_context,
    iterations: int = 5,
    solve_iterations: int = 8,
    damping: float = 1e-4,
):
    """
    Estimate a minimum singular direction of J using
    damped inverse iteration.

    Returns:
        sigma:
            estimated singular value

        u:
            gradient-space left singular direction

        v:
            input-space right singular direction
    """

    if damping <= 0:
        raise ValueError(
            "damping must be greater than zero"
        )

    images, _ = batch_tensors(
        context
    )

    v = torch.randn(
        images.numel(),
        device=images.device,
        dtype=images.dtype,
    )

    norm = torch.linalg.vector_norm(
        v
    )

    if norm == 0:
        raise RuntimeError(
            "Could not initialize min singular direction"
        )

    v = (
        v / norm
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

    sigma = torch.linalg.vector_norm(
        u
    )

    if sigma != 0:
        u = (
            u / sigma
        )

    return sigma, u, v