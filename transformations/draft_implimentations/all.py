import torch


def all_draft(
    ranked: list[torch.Tensor],
    **kwargs
) -> list[torch.Tensor]:

    return [torch.ones_like(score, dtype=torch.bool) for score in ranked]
