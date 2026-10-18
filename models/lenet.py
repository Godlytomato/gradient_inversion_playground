
import torch
import torch.nn as nn

from .model_def import model_definition

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        self.fc1 = nn.LazyLinear(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return self.fc3(x)

#constructor
def build_model() -> nn.Module:
    return LeNet5()

#change funcions
def change_classifier(model: nn.Module, classes: int) -> bool:
    model.fc3 = nn.Linear(model.fc3.in_features, classes)
    return True

def change_channels(model: nn.Module, channels: int) -> bool:
    model.conv1 = nn.Conv2d(
        channels,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
    )
    return True

def change_input(model, dimensions):
    model.fc1 = nn.LazyLinear(120)
    return True

LENET5 = model_definition(
    name="lenet5",
    input_channels=1,
    input_dimensions=(28, 28),
    classes=10,
    model_builder=build_model,
    classifier_change=change_classifier,
    input_change=change_input,
    channel_change=change_channels,
)
