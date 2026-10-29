from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from .dataset_def import dataset_def


def build_mnist():
    return MNIST(root='data', train=False, download=True, transform=ToTensor())


MNIST_DATA = dataset_def(
    name='mnist', dataset_builder=build_mnist, class_count=10,
    input_channels=1, input_dimensions=(28, 28), input_space='raw_pixels', split='test',
    mean=(.1307,), std=(.3081,),
)
