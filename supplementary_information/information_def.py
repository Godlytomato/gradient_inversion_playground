from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class information_def:
    name: str
    provider: Callable
    requires: set[str] = field(default_factory=set)
    parameters: dict[str, Any] = field(default_factory=dict)