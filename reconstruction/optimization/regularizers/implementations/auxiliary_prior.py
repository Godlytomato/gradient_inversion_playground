import torch


def auxiliary_prior_regularizer(
    reconstruction,
    auxiliary_images,
    state=None,
    start_iteration=0
):
    if (
        state is not None
        and state.iteration < start_iteration
    ):
        return reconstruction.new_tensor(0.0)

    if isinstance(auxiliary_images, torch.Tensor):
        references = auxiliary_images

    else:
        references = []

        for sample in auxiliary_images:
            if isinstance(sample, (tuple, list)):
                image = sample[0]
            else:
                image = sample

            references.append(image)

        references = torch.stack(references)

    references = references.to(
        device=reconstruction.device,
        dtype=reconstruction.dtype
    )

    reconstruction_flat = reconstruction.reshape(
        reconstruction.shape[0],
        -1
    )

    references_flat = references.reshape(
        references.shape[0],
        -1
    )

    distances = torch.cdist(
        reconstruction_flat,
        references_flat,
        p=2
    )

    nearest_distance = distances.min(
        dim=1
    ).values

    return nearest_distance.mean()