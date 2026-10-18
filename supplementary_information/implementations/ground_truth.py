import torch

def get_ground_truth(context):
    return torch.stack([
        image for image, _ in context.batch
    ])