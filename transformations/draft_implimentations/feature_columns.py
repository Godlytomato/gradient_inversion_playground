"""Select complete feature columns from one eligible fully connected weight."""

import torch


def feature_columns(scores: list[torch.Tensor], percent: float, **kwargs) -> list[torch.Tensor]:
    if not 0 <= percent <= 1:
        raise ValueError("percent must be between 0 and 1")

    masks = [torch.zeros_like(score, dtype=torch.bool) for score in scores]
    eligible = [index for index, score in enumerate(scores) if torch.isfinite(score).any()]
    if len(eligible) != 1:
        raise ValueError("feature_columns requires exactly one eligible weight tensor")

    index = eligible[0]
    score = scores[index]
    if score.ndim != 2 or not torch.isfinite(score).all():
        raise ValueError("Eligible feature scores must form a finite 2D weight tensor")
    if not torch.allclose(score, score[0:1].expand_as(score)):
        raise ValueError("Each feature column must have a single shared score")

    count = int(score.shape[1] * percent)
    if count:
        columns = torch.topk(score[0], count, largest=True).indices
        masks[index][:, columns] = True
    return masks
