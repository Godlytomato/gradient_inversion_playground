import torch


def lbfgs_optimizer(
    parameters,
    lr=1.0,
    max_iter=20,
    max_eval=None,
    tolerance_grad=1e-7,
    tolerance_change=1e-9,
    history_size=100,
    line_search_fn="strong_wolfe",
):

    return torch.optim.LBFGS(
        parameters,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )