import torch


class direct_gradient_optimizer(torch.optim.Optimizer):

    def __init__(
        self,
        parameters,
        lr=0.1
    ):
        defaults = {
            "lr": lr
        }

        super().__init__(
            parameters,
            defaults
        )

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group["lr"]

            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                parameter.add_(
                    parameter.grad,
                    alpha=-lr
                )