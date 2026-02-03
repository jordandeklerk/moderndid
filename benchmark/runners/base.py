"""Base benchmark runner with common utilities."""

from __future__ import annotations

import gc
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

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
        return asdict(self)


class BaseBenchmarkRunner(ABC):
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

    @abstractmethod
    def time_ddd(
        self,
        data: pl.DataFrame,
        multi_period: bool = False,
        panel: bool = True,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        base_period: str = "varying",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~ cov1 + cov2 + cov3 + cov4",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time DDD estimation."""
