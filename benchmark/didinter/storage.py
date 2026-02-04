"""Result storage for did_multiplegt benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.common.storage import BaseBenchmarkResult


@dataclass
class DIDInterBenchmarkResult(BaseBenchmarkResult):
    """Result from a single did_multiplegt benchmark run."""

    # DIDInter-specific configuration
    n_periods: int = 0
    effects: int = 0
    placebo: int = 0
    normalized: bool = False
