from dataclasses import dataclass
from typing import Callable


@dataclass
class optimization_context:
    objective: Callable