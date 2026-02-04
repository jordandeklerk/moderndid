"""Base benchmark runner with common utilities."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimingResult:
    """Container for timing results from a benchmark run."""

    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    times: list[float]
    n_estimates: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class BaseBenchmarkRunner:
    """Abstract base class for benchmark runners."""

    @staticmethod
    def gc_collect() -> None:
        """Force garbage collection before timing."""
        gc.collect()

    @staticmethod
    def time_execution(func, *args, **kwargs) -> tuple[float, Any]:
        """Time a single function execution."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, result
