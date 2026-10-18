"""Common display conversion for the current raw [0,1] image domain."""
import torch


def display_tensor(image):
    """Clamp for display without per-image contrast changes or normalization."""
    image = image.detach().cpu().float()
    if not torch.isfinite(image).all():
        raise ValueError('Cannot display an image containing nonfinite values')
    return image.clamp(0, 1)


def viewable(image):
    image = display_tensor(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
        if image.shape[-1] == 1:
            image = image[..., 0]
    if image.ndim not in (2, 3):
        raise ValueError('Expected one image with shape [C,H,W] or [H,W]')
    return image.numpy()


def align_for_display(image, image_type, summary):
    key = {'best_reconstruction': 'best_assignment', 'oracle_reconstruction': 'oracle_best_assignment'}.get(image_type)
    assignment = summary.get(key) if key else None
    if image.ndim == 4 and assignment is not None:
        if sorted(assignment) != list(range(len(image))):
            raise ValueError('Invalid stored image assignment')
        return image[assignment]
    return image
