from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass(frozen=True)
class preset_def:

    name: str

    builder: Callable[..., Any] | None = field(
        default=None,
        kw_only=True,
    )

    def build(self, **parameters):

        if self.builder is None:
            raise NotImplementedError(
                f"Preset {self.name} has no builder"
            )

        return self.builder(**parameters)