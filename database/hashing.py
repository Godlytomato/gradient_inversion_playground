"""Versioned canonical identities; historical unprefixed hashes stay distinct."""
import hashlib
import json
from dataclasses import fields, is_dataclass
from collections.abc import Mapping

HASH_VERSION = "v2"


def normalize_config(config):
    if is_dataclass(config) and not isinstance(config, type):
        return {field.name: normalize_config(getattr(config, field.name)) for field in fields(config)}
    if isinstance(config, Mapping):
        if not all(isinstance(key, str) for key in config):
            raise TypeError("Configuration mapping keys must be strings")
        return {key: normalize_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [normalize_config(value) for value in config]
    if config is None or isinstance(config, (str, int, float, bool)):
        return config
    raise TypeError(f"Unsupported configuration value: {type(config).__name__}")


def config_hash(config) -> str:
    encoded = json.dumps(normalize_config(config), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return f"{HASH_VERSION}:" + hashlib.sha256(encoded).hexdigest()


def run_hash(description) -> str:
    return config_hash(description)


def attack_hash(description) -> str:
    return config_hash(description.attack)


def defense_hash(description) -> str:
    return config_hash(description.gradient_transformations)
