import torch
from torch.utils.data import Subset

from data.dataset_def import sample_identity, subset_identities


def auxiliary_images(
    context,
    data_def,
    sample_size=1,
    sampled_classes=None,
    public_data_def=None,
    public_source_disjoint=False,
):
    """Return references that exclude identifiable private dataset records.

    An opaque private tensor/list batch requires an explicitly declared,
    separate disjoint public source, rather than guessing tensor identities.
    """
    source = data_def if public_data_def is None else public_data_def
    explicit_disjoint = public_data_def is not None and public_source_disjoint
    if public_source_disjoint and public_data_def is None:
        raise ValueError("A disjoint public source requires public_data_def")
    if explicit_disjoint and public_data_def is data_def:
        raise ValueError("A disjoint public source cannot be the private dataset definition")
    if (
        source.input_space != data_def.input_space
        or source.input_channels != data_def.input_channels
        or source.input_dimensions != data_def.input_dimensions
    ):
        raise ValueError("Public and private inputs must use the same shape and coordinate system")

    if isinstance(context.batch, Subset):
        private_identities = set(subset_identities(context.batch))
    elif explicit_disjoint:
        private_identities = set()
    else:
        raise ValueError(
            "Auxiliary sampling requires private Subset identities or an explicitly disjoint public source"
        )

    if not explicit_disjoint and len(source.data):
        source_base = sample_identity(source.data, 0)[0]
        if any(identity[0] != source_base for identity in private_identities):
            raise ValueError(
                "Private and public dataset identities cannot be compared; declare a disjoint public source explicitly"
            )

    if sampled_classes is None:
        sampled_classes = list(range(source.class_count))
    samples = source.sample(
        sample_size=sample_size,
        sampled_classes=sampled_classes,
        exclude_identities=private_identities,
    )
    public_identities = subset_identities(samples)
    if private_identities.intersection(public_identities):
        raise ValueError("Public auxiliary examples overlap the private batch")

    metadata = getattr(context, "experiment_metadata", None)
    if metadata is None:
        metadata = {}
        context.experiment_metadata = metadata
    metadata["auxiliary"] = {
        "dataset": source.name,
        "split": source.split,
        "input_space": source.input_space,
        "base_indices": [index for _, index in public_identities],
        "excluded_private_base_indices": sorted(index for _, index in private_identities),
        "sampling_with_replacement": True,
        "explicit_disjoint_source": bool(explicit_disjoint),
    }

    parameter = next(context.model.parameters())
    return torch.stack([image for image, _ in samples]).to(
        device=parameter.device, dtype=parameter.dtype
    )
