import torch


def scale_loss(attack_loss, scale, state=None):
    return attack_loss * scale
