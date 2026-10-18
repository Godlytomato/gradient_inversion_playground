import torch


def scale_loss(loss,scale,state=None):
    return loss * scale