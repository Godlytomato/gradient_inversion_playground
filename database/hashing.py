import hashlib
import json

from dataclasses import asdict, is_dataclass


def normalize_config(config):
    """
    Convert configuration objects into a deterministic
    JSON-serializable structure.
    """

    if is_dataclass(config):
        config = asdict(config)

    return config


def config_hash(config) -> str:
    """
    Create a deterministic SHA256 hash for a configuration.
    """

    normalized = normalize_config(config)

    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def run_hash(description) -> str:
    """
    Hash the complete run description.
    """

    return config_hash(
        description
    )


def attack_hash(description) -> str:
    """
    Hash only the reconstruction attack configuration.
    """

    return config_hash(
        description.attack
    )


def defense_hash(description) -> str:
    """
    Hash only the gradient transformation configuration.
    """

    return config_hash(
        description.gradient_transformations
    )