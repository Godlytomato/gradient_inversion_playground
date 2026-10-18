import torch

from run_utils.runtime_context import runtime_context


def get_labels(context: runtime_context) -> torch.Tensor:
    labels = [label for _, label in context.batch]
    return torch.as_tensor(
        labels, dtype=torch.long, device=next(context.model.parameters()).device
    )
