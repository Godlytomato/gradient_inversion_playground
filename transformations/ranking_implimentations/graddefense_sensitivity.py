import torch

from transformations.transform_def import (
    grad_package,
    transformation_context,
)


def graddefense_sensitivity(
    package: grad_package,
    context: transformation_context,
    **kwargs
) -> list[torch.Tensor]:

    model = context.model
    batch = context.batch

    # build tensors from dataset/subset batches
    if isinstance(batch, torch.Tensor):
        inputs = batch
        labels = context.labels

    else:
        images = []
        labels_list = []

        for sample in batch:

            if not isinstance(
                sample,
                (tuple, list)
            ):
                raise TypeError(
                    "GradDefense requires batch samples "
                    "to contain image and label"
                )

            image, label = sample

            if not isinstance(
                image,
                torch.Tensor
            ):
                raise TypeError(
                    "GradDefense requires image tensors"
                )

            images.append(
                image
            )

            labels_list.append(
                label
            )

        if not images:
            raise ValueError(
                "GradDefense received an empty batch"
            )

        inputs = torch.stack(
            images
        )

        labels = torch.tensor(
            labels_list,
            dtype=torch.long,
        )


    # context labels can override extracted labels
    if context.labels is not None:
        labels = context.labels


    if labels is None:
        raise ValueError(
            "GradDefense sensitivity requires labels"
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

    labels = labels.to(
        device
    )


    loss_fn = context.loss_function

    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()


    model.zero_grad(
        set_to_none=True
    )


    predictions = model(
        inputs
    )


    loss = loss_fn(
        predictions,
        labels
    )


    gradients = torch.autograd.grad(
        outputs=loss,
        inputs=model.parameters(),
        create_graph=True,
        allow_unused=True,
    )


    scores = []

    for gradient, package_gradient in zip(
        gradients,
        package.grads
    ):

        if gradient is None:

            scores.append(
                torch.zeros_like(
                    package_gradient
                )
            )

            continue


        input_vjp = torch.autograd.grad(
            outputs=gradient,
            inputs=inputs,
            grad_outputs=torch.ones_like(
                gradient
            ),
            retain_graph=True,
            allow_unused=True,
        )[0]


        if input_vjp is None:

            sensitivity = torch.tensor(
                0.0,
                device=package_gradient.device,
            )

        else:

            sample_norms = torch.norm(
                input_vjp.reshape(
                    input_vjp.shape[0],
                    -1,
                ),
                dim=1,
            )

            sensitivity = (
                sample_norms.mean()
            )


        scores.append(
            torch.full_like(
                package_gradient,
                sensitivity.detach(),
            )
        )


    return scores