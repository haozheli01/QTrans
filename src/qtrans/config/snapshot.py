"""Create private configuration state for QTrans objects."""

from collections.abc import Mapping
from copy import deepcopy


def snapshot_configuration(config):
    """Return a deep, mutable copy of a user configuration mapping."""
    if not isinstance(config, Mapping):
        raise TypeError("QTrans configuration must be a mapping.")
    return deepcopy(dict(config))


def apply_defaults(target, defaults):
    """Copy default values onto an object without sharing mutable values."""
    for parameters in defaults.values():
        for name, value in parameters.items():
            setattr(target, name, deepcopy(value))

