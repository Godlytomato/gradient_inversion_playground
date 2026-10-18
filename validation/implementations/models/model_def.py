from dataclasses import dataclass

from validation.implementations.preset_def import preset_def
from models.model_registry import model_dict


@dataclass(frozen=True)
class model_preset_def(preset_def):

    # name used by the execution model registry
    model_name: str

    def build(self) -> str:

        if self.model_name not in model_dict:
            raise ValueError(
                f"Unknown model: {self.model_name}"
            )

        return self.model_name