import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)


def soteria_sensitivity(
    package: grad_package,
    context: transformation_context,
    layer_name: str = "auto",
    stabilizer: float = 0.1,
) -> list[torch.Tensor]:

    model = context.model
    batch = context.batch

    # context.batch is normally a dataset/subset containing
    # (image, label) samples.
    if isinstance(batch, torch.Tensor):
        inputs = batch

    else:
        images = []

        for sample in batch:

            if isinstance(
                sample,
                (tuple, list)
            ):
                image = sample[0]
            else:
                image = sample

            if not isinstance(
                image,
                torch.Tensor
            ):
                raise TypeError(
                    "Soteria requires batch samples "
                    "to contain image tensors"
                )

            images.append(
                image
            )

        if not images:
            raise ValueError(
                "Soteria received an empty batch"
            )

        inputs = torch.stack(
            images
        )

    device = next(
        model.parameters()
    ).device

    inputs = (
        inputs
        .detach()
        .clone()
        .to(device)
    )

    inputs.requires_grad_(
        True
    )

    modules = dict(
        model.named_modules()
    )

    if layer_name == "auto":
        candidates = [name for name, module in modules.items() if isinstance(module, torch.nn.Linear)]
        if not candidates:
            raise ValueError("Soteria requires a model with a Linear layer")
        layer_name = candidates[-1]

    if layer_name not in modules:
        raise ValueError(
            f"Unknown Soteria layer: {layer_name}"
        )

    layer = modules[
        layer_name
    ]

    if not isinstance(
        layer,
        torch.nn.Linear
    ):
        raise TypeError(
            "Soteria currently expects the defended layer "
            "to be torch.nn.Linear"
        )

    representation = None

    def capture_representation(
        module,
        args,
        output
    ):
        nonlocal representation

        representation = args[0]

    hook = layer.register_forward_hook(
        capture_representation
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        model(
            inputs
        )

    finally:
        hook.remove()

    if representation is None:
        raise RuntimeError(
            "Soteria could not capture the layer representation"
        )

    representation = representation.reshape(
        representation.shape[0],
        -1
    )

    feature_count = (
        representation.shape[1]
    )

    sensitivity = torch.zeros(
        feature_count,
        device=representation.device,
        dtype=representation.dtype,
    )

    for feature_index in range(
        feature_count
    ):

        target = torch.zeros_like(
            representation
        )

        target[
            :,
            feature_index
        ] = 1

        input_gradient = torch.autograd.grad(
            outputs=representation,
            inputs=inputs,
            grad_outputs=target,
            retain_graph=True,
            create_graph=False,
        )[0]

        derivative_norm = torch.norm(
            input_gradient.reshape(
                input_gradient.shape[0],
                -1
            ),
            dim=1,
        )

        feature_value = representation[
            :,
            feature_index
        ]

        feature_score = (
            derivative_norm
            /
            (
                feature_value.abs()
                + stabilizer
            )
        )

        sensitivity[
            feature_index
        ] = feature_score.sum()

    scores = [
        torch.full_like(
            gradient,
            float("-inf")
        )
        for gradient in package.grads
    ]

    parameters = list(
        model.parameters()
    )

    weight_index = None

    for index, parameter in enumerate(
        parameters
    ):
        if parameter is layer.weight:
            weight_index = index
            break

    if weight_index is None:
        raise RuntimeError(
            "Could not match the Soteria layer weight "
            "to its gradient"
        )

    weight_gradient = package.grads[
        weight_index
    ]

    if weight_gradient.ndim != 2:
        raise ValueError(
            "Soteria currently expects a 2D FC weight gradient"
        )

    if weight_gradient.shape[1] != feature_count:
        raise ValueError(
            "Soteria representation dimension does not match "
            "the defended layer input dimension"
        )

    scores[
        weight_index
    ] = (
        sensitivity
        .reshape(1, -1)
        .expand_as(weight_gradient)
    )

    return scores
