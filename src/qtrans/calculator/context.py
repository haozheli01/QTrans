"""Typed boundaries for future composed calculator services."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

from qtrans.core.context import ParallelContext


@dataclass
class CalculationContext:
    """Shared inputs and state passed between calculation services."""

    config: Mapping[str, Any]
    parallel: ParallelContext
    state: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalculationResult:
    """Common envelope for a calculation mode result."""

    mode: str
    values: Mapping[str, Any]
    output_directory: Optional[str] = None

