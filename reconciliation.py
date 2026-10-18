
from data.dataset_def import dataset_def
from models.model_def import model_definition

import torch.nn as nn


def reconciliation(
    model_def: model_definition,
    data: dataset_def,
    model: nn.Module,
):
    if model_def.input_channels != data.input_channels:
        if not model_def.channel_available:
            raise ValueError(
                f"{model_def.name} cannot adapt input channels"
            )

        model_def.channel_change(model, data.input_channels)

    if model_def.input_dimensions != data.input_dimensions:
        if model_def.input_available:
            model_def.input_change(model, data.input_dimensions)
        # Otherwise preserve the model's existing architecture.

    if model_def.classes != data.class_count:
        if not model_def.class_available:
            raise ValueError(
                f"{model_def.name} cannot adapt class count"
            )

        model_def.classifier_change(model, data.class_count)

    return model
