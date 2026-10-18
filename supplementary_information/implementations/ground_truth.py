import torch

def get_ground_truth(context):
    parameter = next(context.model.parameters())
    return torch.stack([
        image for image, _ in context.batch
    ]).to(device=parameter.device, dtype=parameter.dtype)
