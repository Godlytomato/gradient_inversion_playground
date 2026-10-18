from reconstruction.optimization.labels.label_policy_def import (
    label_policy_def,
)
from reconstruction.optimization.labels.implementations.unknown import (
    unknown_labels,
)
from reconstruction.optimization.labels.implementations.known import (
    known_labels,
)
from reconstruction.optimization.labels.implementations.idlg import (
    idlg_labels,
)
from reconstruction.optimization.labels.implementations.hf_gradinv_labls import (
    hf_gradinv_labels,
)
from reconstruction.optimization.labels.implementations.hf_gradinv_labls import (
    hf_gradinv_labels,
)

label_policy_registry = {

    "hf_gradinv": label_policy_def(
        name="hf_gradinv",
        function=hf_gradinv_labels,
        requires={
            "model",
            "gradients",
            "auxiliary_images",
            "batch_size",
        },
    ),

    "known": label_policy_def(
        name="known",
        function=known_labels,
        requires={
            "labels",
        },
    ),
    "unknown": label_policy_def(
        name="unknown",
        function=unknown_labels,
        requires={
            "batch_size",
            "class_count",
            "device",
        },
    ),
    "idlg": label_policy_def(
        name="idlg",
        function=idlg_labels,
        requires={
            "gradients",
        },
    ),
    "hf_gradinv": label_policy_def(
        name="hf_gradinv",
        function=hf_gradinv_labels,
        requires={
            "model",
            "gradients",
            "auxiliary_images",
            "batch_size",
        },
    ),
}