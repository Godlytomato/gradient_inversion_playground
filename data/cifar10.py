from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from .dataset_def import dataset_def

def build_cifar10():
    return CIFAR10(root="data",train=False,download=True,transform=ToTensor())

CIFAR10_DATA = dataset_def(
    name="cifar10",

    dataset_builder=build_cifar10,
    class_count=10,
    input_channels=3,
    input_dimensions=(32, 32),
    input_space="raw_pixels",
    split="test",
    mean=(
        0.4914,
        0.4822,
        0.4465
    ),

    std=(
        0.2470,
        0.2435,
        0.2616
    ),
)
