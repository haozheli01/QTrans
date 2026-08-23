"""Load configuration files and establish independent configuration state."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path


def load_configuration(path):
    """Load a YAML configuration mapping from ``path``."""
    import yaml

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("QTrans configuration must be a YAML mapping.")
    return data


def copy_configuration(config):
    """Return a deep, mutable copy of a user configuration mapping."""
    if not isinstance(config, Mapping):
        raise TypeError("QTrans configuration must be a mapping.")
    return deepcopy(dict(config))


def apply_defaults(target, defaults):
    """Copy default values onto an object without sharing mutable values."""
    for parameters in defaults.values():
        for name, value in parameters.items():
            setattr(target, name, deepcopy(value))
