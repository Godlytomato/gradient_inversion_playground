import torch.nn as nn
from torchvision import models

from .model_def import model_definition


def build_model() -> nn.Module:
    return models.resnet18(
        weights=None
    )


def change_classifier(
    model: nn.Module,
    classes: int
) -> bool:

    model.fc = nn.Linear(
        model.fc.in_features,
        classes,
    )

    return True


def change_channels(
    model: nn.Module,
    channels: int
) -> bool:

    old = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )

    return True


def change_input(
    model: nn.Module,
    dimensions: tuple[int, int]
) -> bool:
    """
    Adapt the ImageNet ResNet stem for small images such as
    CIFAR-10.

    Standard ResNet uses a 7x7 stride-2 convolution followed by
    max pooling. On 32x32 images this reduces the spatial
    dimensions too aggressively, eventually producing 1x1
    feature maps.

    For small images, use the common CIFAR-style stem:
        - 3x3 convolution
        - stride 1
        - no initial max pooling
    """

    height, width = dimensions

    if height <= 64 and width <= 64:

        old = model.conv1

        model.conv1 = nn.Conv2d(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()

    return True


RESNET18 = model_definition(
    name="resnet18",
    input_channels=3,
    input_dimensions=(224, 224),
    classes=1000,
    model_builder=build_model,
    classifier_change=change_classifier,
    input_change=change_input,
    channel_change=change_channels,
)