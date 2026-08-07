"""Shared execution context contracts."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ParallelContext:
    """MPI topology made explicit for future composed calculation services."""

    communicator: Optional[Any]
    rank: int
    size: int
    nodes: int
    minimum_tasks_per_node: int

