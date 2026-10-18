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

        # split equally between highest and lowest
        each_side = selected // 2

        mask = torch.zeros_like(
            flat,
            dtype=torch.bool
        )

        if each_side > 0:

            order = torch.argsort(flat)

            lowest = order[:each_side]
            highest = order[-each_side:]

            mask[lowest] = True
            mask[highest] = True

        masks.append(
            mask.reshape_as(score)
        )

    return masks