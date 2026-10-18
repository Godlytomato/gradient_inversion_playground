import torch


def tv_regularizer(
    reconstruction,
    state=None,
    start_iteration=0
):

    if (
        state is not None
        and state.iteration < start_iteration
    ):
        return reconstruction.new_tensor(0.0)

    horizontal = torch.abs(
        reconstruction[:, :, :, 1:]
        - reconstruction[:, :, :, :-1]
    ).mean()

    vertical = torch.abs(
        reconstruction[:, :, 1:, :]
        - reconstruction[:, :, :-1, :]
    ).mean()

    return horizontal + vertical