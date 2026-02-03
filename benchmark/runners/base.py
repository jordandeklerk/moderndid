"""Base benchmark runner with common utilities."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl


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
        return {
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "times": self.times,
            "n_estimates": self.n_estimates,
            "success": self.success,
            "error": self.error,
        }


class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    @staticmethod
    def gc_collect() -> None:
        """Force garbage collection before timing."""
        gc.collect()

    @staticmethod
    def time_execution(func, *args, **kwargs) -> tuple[float, any]:
        """Time a single function execution."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return elapsed, result

    @abstractmethod
    def time_att_gt(
        self,
        data: pl.DataFrame,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~1",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time att_gt estimation."""
