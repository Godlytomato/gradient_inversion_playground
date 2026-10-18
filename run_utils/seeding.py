"""Independent, reproducible random streams for experimental stages."""
import hashlib
import random

import numpy as np
import torch


def stage_seed(experiment_seed: int, stage: str) -> int:
    encoded = f"gradient-playground-v2:{experiment_seed}:{stage}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
