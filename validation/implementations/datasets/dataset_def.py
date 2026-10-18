from dataclasses import dataclass

from run_def import batch
from validation.implementations.preset_def import preset_def


@dataclass(frozen=True)
class dataset_preset_def(preset_def):

    dataset_name: str
    batch_size: int = 1
    classes: tuple[int, ...] | None = None

    def build(self) -> batch:

        from data.dataset_registry import dataset_dict

        if self.dataset_name not in dataset_dict:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}"
            )

        dataset = dataset_dict[self.dataset_name]

        if self.batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero"
            )

        if self.classes is None:
            classes = list(range(dataset.class_count))
        else:
            classes = list(self.classes)

        if not classes:
            raise ValueError(
                "At least one class must be selected"
            )

        if any(
            class_id < 0 or class_id >= dataset.class_count
            for class_id in classes
        ):
            raise ValueError(
                "Selected classes are outside the dataset range"
            )

        return batch(
            dataset=self.dataset_name,
            size=self.batch_size,
            classes=classes,
        )