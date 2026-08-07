"""Typed boundaries for future composed preparer services."""

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

from qtrans.core.context import ParallelContext


@dataclass
class PreparationContext:
    """Shared inputs and state passed between preparation services."""

    config: Mapping[str, Any]
    parallel: ParallelContext
    state: MutableMapping[str, Any] = field(default_factory=dict)

