import torch


class bn_stat_hook:

    def __init__(self, module):
        self.module = module
        self.mean = None
        self.var = None

        self.hook = module.register_forward_hook(
            self._hook
        )

    def _hook(
        self,
        module,
        inputs,
        output
    ):
        activation = inputs[0]

        dimensions = [
            0
        ] + list(
            range(2, activation.dim())
        )

        self.mean = activation.mean(
            dim=dimensions
        )

        self.var = activation.var(
            dim=dimensions,
            unbiased=False
        )

    def remove(self):
        self.hook.remove()


def bn_stats_regularizer(
    reconstruction,
    model,
    state=None,
    start_iteration=0
):
    # regularizer has not started yet
    if (
        state is not None
        and state.iteration < start_iteration
    ):
        return reconstruction.new_tensor(0.0)

    hooks = []

    # collect batch norm layers
    for module in model.modules():

        if isinstance(
            module,
            torch.nn.modules.batchnorm._BatchNorm
        ):
            hooks.append(
                bn_stat_hook(module)
            )

    # model has no batch norm
    # so this regularizer contributes nothing
    if not hooks:
        return reconstruction.new_tensor(0.0)

    # forward pass so hooks capture activation stats
    model(reconstruction)

    loss = reconstruction.new_tensor(0.0)

    for hook in hooks:

        if (
            hook.mean is None
            or hook.var is None
        ):
            continue

        # skip batch norm layers that do not track
        # running statistics
        if (
            hook.module.running_mean is None
            or hook.module.running_var is None
        ):
            continue

        mean_loss = torch.mean(
            (
                hook.mean
                - hook.module.running_mean
            ) ** 2
        )

        var_loss = torch.mean(
            (
                hook.var
                - hook.module.running_var
            ) ** 2
        )

        loss = (
            loss
            + mean_loss
            + var_loss
        )

    for hook in hooks:
        hook.remove()

    return loss