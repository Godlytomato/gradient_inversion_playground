from .lenet import LENET5
from .resnet18 import RESNET18
from .resnet50 import RESNET50


model_dict = {
    LENET5.name: LENET5,
    RESNET18.name: RESNET18,
    RESNET50.name: RESNET50,
}