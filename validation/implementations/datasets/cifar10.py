from validation.implementations.datasets.dataset_def import (
    dataset_preset_def,
)


CIFAR10_PRESET = dataset_preset_def(
    name="cifar10",
    dataset_name="cifar10",
    batch_size=1,
    classes=None,
)