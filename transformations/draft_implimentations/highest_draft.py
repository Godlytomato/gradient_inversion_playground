import torch

from .draft_utils import (
    flatten_scores,
    selection_count,
    reshape_mask
)


def highest_draft(
    scores: list[torch.Tensor],
    percent: float,
    participants: int = 1,
    participant_index: int = 0
) -> list[torch.Tensor]:

    if participants < 1:
        raise ValueError(
            "participants must be at least 1"
        )

    if participant_index < 0 or participant_index >= participants:
        raise ValueError(
            "participant_index must be between "
            "0 and participants - 1"
        )

    flat_scores = flatten_scores(scores)

    count = selection_count(
        scores,
        percent
    )

    flat_mask = torch.zeros_like(
        flat_scores,
        dtype=torch.bool
    )

    if count == 0:
        return reshape_mask(
            flat_mask,
            scores
        )

    # rank highest first
    ranked_indices = torch.topk(
        flat_scores,
        k=count,
        largest=True
    ).indices

    # simulate multiple participants taking turns
    selected_indices = ranked_indices[
        participant_index::participants
    ]

    flat_mask[selected_indices] = True

    return reshape_mask(
        flat_mask,
        scores
    )