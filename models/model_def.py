from dataclasses import dataclass, field
from typing import Callable
import torch.nn as nn

@dataclass
class model_definition:

    name: str

    #reconciliation stuff
    input_channels: int
    input_dimensions: tuple[int, int]
    classes:int

    #thing that makes the model, Callable is just function this is for somthing that retursn us module
    model_builder:Callable[[],nn.Module]


    #things that do the adaptation
    #takes an arg and nn and returnts true if succsefull or false on fail and ill make somt flag to see if it fail
    #=None defaults it to None
    classifier_change: Callable[[nn.Module,int],bool]
    input_change: Callable[[nn.Module,tuple[int,int]],bool]
    channel_change: Callable[[nn.Module,int],bool]

    #protpety is just a like classfunction
    #callable is a function that returns a bool if its callable
    @property
    def class_available(self)->bool:
        return callable(self.classifier_change)
    @property
    def input_available(self)->bool:
        return callable(self.input_change)
    @property
    def channel_available(self)->bool:
        return callable(self.channel_change)