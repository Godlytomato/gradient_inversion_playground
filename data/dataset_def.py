from dataclasses import dataclass, field
from typing import Callable, Any
import random
from torch.utils.data import Dataset,Subset




@dataclass
class dataset_def:
    name:str

    dataset_builder:Callable[[],Dataset]

    class_count: int
    input_channels: int
    input_dimensions: tuple[int, int]

    mean: tuple[float, ...] | None = None
    std: tuple[float, ...] | None = None

    data: Dataset = field(init=False)
    #this defines my data for me the post init thing means it runs after init
    def __post_init__(self):
        self.data = self.dataset_builder()
    
    #this is simple if you forget how to type your sad liam
    def sample(self,sample_size:int,sampled_classes:list[int])->Subset:
        selected = []
        while len(selected)<sample_size:
            #select
            index = random.randrange(len(self.data))
            #retreive
            _,label=self.data[index]

            if int(label) in sampled_classes:
                selected.append(index)

        return Subset(self.data,selected)