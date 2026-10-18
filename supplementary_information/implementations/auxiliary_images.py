import torch


def auxiliary_images(
    context,
    data_def,
    sample_size=1,
    sampled_classes=None
):
    if sampled_classes is None:
        sampled_classes = list(
            range(data_def.class_count)
        )

    samples = data_def.sample(
        sample_size=sample_size,
        sampled_classes=sampled_classes
    )

    images = [
        image
        for image, _ in samples
    ]

    return torch.stack(images)