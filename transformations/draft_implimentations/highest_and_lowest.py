import torch


def highest_lowest(
    scores: list[torch.Tensor],
    percent: float,
    **kwargs
) -> list[torch.Tensor]:

    if percent < 0 or percent > 1:
        raise ValueError(
            "percent must be between 0 and 1"
        )

    masks = []

    for score in scores:

        flat = score.reshape(-1)

        total = flat.numel()

        # total fraction selected across both ends
        selected = int(total * percent)

        # Put an odd remainder on the high end, preserving the exact count.
        lowest_count = selected // 2
        highest_count = selected - lowest_count

        mask = torch.zeros_like(
            flat,
            dtype=torch.bool
        )

        if selected > 0:

            order = torch.argsort(flat)

            lowest = order[:lowest_count]
            highest = order[-highest_count:]

            mask[lowest] = True
            mask[highest] = True

        masks.append(
            mask.reshape_as(score)
        )

    return masks
