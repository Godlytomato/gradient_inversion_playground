from dataclasses import dataclass
from typing import Any

@dataclass
class required_information:
    information: dict[str, Any]

    def get(self, name: str):
        return self.information[name]

    def has(self, name: str) -> bool:
        return name in self.information