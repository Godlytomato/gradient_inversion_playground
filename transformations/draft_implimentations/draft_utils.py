# transformations/draft_implimentations/draft_utils.py

import torch


def flatten_scores(
    scores: list[torch.Tensor]
) -> torch.Tensor:
    flattened = torch.cat([
        score.reshape(-1)
        for score in scores
    ])
    if not torch.isfinite(flattened).all():
        raise ValueError(
            "Coordinate drafts require finite scores; use feature_columns "
            "for Soteria eligibility scores and regenerate historical presets"
        )
    return flattened


def selection_count(
    scores: list[torch.Tensor],
    percent: float
) -> int:

    if percent < 0 or percent > 1:
        raise ValueError(
            f"percent must be between 0 and 1, got {percent}"
        )

    total = sum(
        score.numel()
        for score in scores
    )

    return int(total * percent)


def reshape_mask(
    flat_mask: torch.Tensor,
    scores: list[torch.Tensor]
) -> list[torch.Tensor]:

    masks = []
    position = 0

    for score in scores:
        size = score.numel()

        mask = flat_mask[
            position:position + size
        ].reshape(score.shape)

        masks.append(mask)

        position += size

    return masks
