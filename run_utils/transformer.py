import torch

from run_def import grad_transform
from transformations.transform_def import (
    grad_package,
    transformation_context
)

from transformations.transform_registry import (
    transform_dict,
    noise_construction_dict,
    noise_scaling_dict
)

from transformations.application_selection_policy_registrys import (
    ranking_policy_dict,
    draft_policy_dict
)


def transformer(
    grads: list[torch.Tensor],
    transformation_list: list[grad_transform],
    context: transformation_context
) -> grad_package:

    package = grad_package(
        grads=grads,
        missing_mask=None
    )

    for transform_desc in transformation_list:

        # ranking
        ranking_desc = transform_desc.ranking_policy
        ranking = ranking_policy_dict[ranking_desc.name]

        scores = ranking.func(
            package,
            context,
            **ranking_desc.parameters
        )

        # draft
        draft_desc = transform_desc.draft_policy
        draft = draft_policy_dict[draft_desc.name]

        selection_masks = draft.func(
            scores,
            **draft_desc.parameters
        )

        # transformation
        transform = transform_desc.transform

        if transform.name == "zero":

            zero_transform = transform_dict["zero"]

            package = zero_transform.func(
                package,
                selection_masks
            )

        elif transform.name == "noise":

            construction_desc = transform.parameters["construction"]
            scaling_desc = transform.parameters["scaling"]

            # construct noise
            construction = noise_construction_dict[
                construction_desc["name"]
            ]

            noise_tensors = construction.func(
                package,
                selection_masks,
                context,
                **construction_desc.get("parameters", {})
            )

            # scale noise
            scaling = noise_scaling_dict[
                scaling_desc["name"]
            ]

            noise_tensors = scaling.func(
                package,
                noise_tensors,
                selection_masks,
                context,
                **scaling_desc.get("parameters", {})
            )

            # apply noise
            noise_transform = transform_dict["noise"]

            package = noise_transform.func(
                package,
                noise_tensors,
                selection_masks
            )

        else:
            raise ValueError(
                f"Unknown transformation: {transform.name}"
            )

    return package