import torch
import torch.nn as nn
from torch.utils.data import Dataset

def grad_prod(model:nn.Module,batch:Dataset)->list[torch.Tensor]:
    model.zero_grad()
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    forward = model(images)

    loss = nn.CrossEntropyLoss()(forward, labels)
    loss.backward()
    
    #add back into shape
    grads = []
    for p in model.parameters():
        grads.append(p.grad.clone())

    return grads