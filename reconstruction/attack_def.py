from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class component_def:
    name: str
    function: Callable
    requires: set[str] = field(default_factory=set)
    parameter_validator: Callable[[dict[str, Any]], None] | None = None


@dataclass
class component_slot:
    registry: dict[str, component_def]
    required: bool = False
    multiple: bool = False


@dataclass
class attack_def:
    name: str
    function: Callable
    requires: set[str] = field(default_factory=set)
    components: dict[str, component_slot] = field(default_factory=dict)
    validator: Callable | None = None