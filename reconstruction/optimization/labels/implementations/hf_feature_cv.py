import torch


def find_feature_layer(model):

    feature_layer = None

    # find the last linear layer in the model
    for module in model.modules():

        if isinstance(
            module,
            torch.nn.Linear
        ):
            feature_layer = module

    if feature_layer is None:
        raise ValueError(
            f"Could not find a Linear feature layer "
            f"for model {type(model).__name__}"
        )

    return feature_layer


def collect_features(
    model,
    auxiliary_images
):

    feature_layer = find_feature_layer(
        model
    )

    collected = []

    # capture the INPUT to the final classifier
    def hook(
        module,
        inputs,
        output
    ):

        features = inputs[0]

        collected.append(
            features.detach()
        )

    handle = feature_layer.register_forward_hook(
        hook
    )

    device = next(
        model.parameters()
    ).device

    with torch.no_grad():

        for sample in auxiliary_images:

            # dataset samples may be (image, label)
            if isinstance(
                sample,
                (tuple, list)
            ):
                sample = sample[0]

            # add batch dimension
            if sample.dim() == 3:
                sample = sample.unsqueeze(0)

            sample = sample.to(
                device
            )

            model(sample)

    handle.remove()

    if len(collected) == 0:
        raise ValueError(
            "No auxiliary features were collected"
        )

    return torch.cat(
        collected,
        dim=0
    )


def coefficient_of_variation(
    features,
    eps=1e-8
):

    mean = features.mean(
        dim=0
    )

    std = features.std(
        dim=0,
        unbiased=False
    )

    return std / (
        mean.abs()
        + eps
    )