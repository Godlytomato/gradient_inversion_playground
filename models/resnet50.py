import torch.nn as nn
from torchvision import models

from .model_def import model_definition


def build_model() -> nn.Module:
    return models.resnet50(weights=None)


def change_classifier(model: nn.Module, classes: int) -> bool:
    model.fc = nn.Linear(
        model.fc.in_features,
        classes,
    )
    return True


def change_channels(model: nn.Module, channels: int) -> bool:
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


RESNET50 = model_definition(
    name="resnet50",
    input_channels=3,
    input_dimensions=(224, 224),
    classes=1000,
    model_builder=build_model,
    classifier_change=change_classifier,
    input_change=None,
    channel_change=change_channels,
    supports_variable_input=True,
)
