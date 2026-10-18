from dataclasses import dataclass

@dataclass
class optimization_state:
    iteration: int
    restart: int
    learning_rate: float | None = None