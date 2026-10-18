from supplementary_information.information_def import information_def

from supplementary_information.implementations.model import get_model
from supplementary_information.implementations.gradients import get_gradients
from supplementary_information.implementations.batch_size import get_batch_size
from supplementary_information.implementations.labels import get_labels
from supplementary_information.implementations.input_shape import get_input_shape
from supplementary_information.implementations.class_count import get_class_count
from supplementary_information.implementations.ground_truth import get_ground_truth
from supplementary_information.implementations.device import get_device
from supplementary_information.implementations.normalization import (
    normalization as get_normalization,
)
from supplementary_information.implementations.auxiliary_images import (
    auxiliary_images,
)
from supplementary_information.implementations.data_def import get_data_def


information_registry = {
    "data_def": information_def(
        name="data_def",
        provider=get_data_def,
    ),

    "model": information_def(
        name="model",
        provider=get_model,
    ),

    "gradients": information_def(
        name="gradients",
        provider=get_gradients,
    ),

    "batch_size": information_def(
        name="batch_size",
        provider=get_batch_size,
    ),

    "labels": information_def(
        name="labels",
        provider=get_labels,
    ),

    "input_shape": information_def(
        name="input_shape",
        provider=get_input_shape,
    ),

    "class_count": information_def(
        name="class_count",
        provider=get_class_count,
    ),

    "ground_truth": information_def(
        name="ground_truth",
        provider=get_ground_truth,
    ),

    "device": information_def(
        name="device",
        provider=get_device,
    ),

    "normalization": information_def(
        name="normalization",
        provider=get_normalization,
        requires={"data_def"},
    ),

    "auxiliary_images": information_def(
        name="auxiliary_images",
        provider=auxiliary_images,
        requires={"data_def"},
        parameters={
            "sample_size": 100,
        },
    ),
}