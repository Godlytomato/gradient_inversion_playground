from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Collection
import random

from torch.utils.data import Dataset, Subset


def sample_identity(dataset: Dataset, index: int) -> tuple[int, int]:
    """Resolve nested subsets to an identity in their original dataset."""
    while True:
        index = int(index)
        if index < 0:
            index += len(dataset)
        if index < 0 or index >= len(dataset):
            raise IndexError("Sample identity index is outside its dataset")
        if not isinstance(dataset, Subset):
            return id(dataset), index
        index = dataset.indices[index]
        dataset = dataset.dataset


def subset_identities(samples: Subset) -> list[tuple[int, int]]:
    if not isinstance(samples, Subset):
        raise ValueError("Sample identities require a torch.utils.data.Subset")
    return [sample_identity(samples.dataset, int(index)) for index in samples.indices]


def _label_at(dataset: Dataset, index: int) -> int:
    # Resolve Subsets before reading CIFAR's labels without image transforms.
    while isinstance(dataset, Subset):
        index = int(dataset.indices[index])
        dataset = dataset.dataset
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return int(targets[index])
    return int(dataset[index][1])


@dataclass
class dataset_def:
    name: str
    dataset_builder: Callable[[], Dataset]
    class_count: int
    input_channels: int
    input_dimensions: tuple[int, int]
    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None
    input_space: str = "raw_pixels"
    split: str | None = None

    _data: Dataset | None = field(init=False, default=None, repr=False)

    @property
    def data(self) -> Dataset:
        """Build data on first access, allowing offline imports/configuration."""
        if self._data is None:
            self._data = self.dataset_builder()
        return self._data

    @data.setter
    def data(self, value: Dataset) -> None:
        # Preserve existing fixtures and local dataset overrides.
        self._data = value

    def sample(
        self,
        sample_size: int,
        sampled_classes: list[int],
        *,
        replace: bool = True,
        exclude_identities: Collection[tuple[int, int]] = (),
    ) -> Subset:
        """Sample a finite eligible population, with replacement by default."""
        if isinstance(sample_size, bool) or not isinstance(sample_size, Integral) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean")
        if not sampled_classes:
            raise ValueError("At least one class must be selected")
        if any(
            isinstance(label, bool)
            or not isinstance(label, Integral)
            or label < 0
            or label >= self.class_count
            for label in sampled_classes
        ):
            raise ValueError("Selected classes are outside the dataset range")

        classes = set(sampled_classes)
        excluded = set(exclude_identities)
        data = self.data
        eligible = [
            index for index in range(len(data))
            if sample_identity(data, index) not in excluded
            and _label_at(data, index) in classes
        ]
        if not eligible:
            raise ValueError("No eligible examples remain for the selected classes and exclusions")
        if not replace and sample_size > len(eligible):
            raise ValueError("sample_size exceeds the eligible population without replacement")

        selected = (
            random.choices(eligible, k=int(sample_size))
            if replace else random.sample(eligible, k=int(sample_size))
        )
        return Subset(data, selected)
