"""Result storage for ddd benchmark outputs."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.common.storage import BaseBenchmarkResult


@dataclass
class DDDBenchmarkResult(BaseBenchmarkResult):
    """Container for DDD benchmark result."""

    dgp_type: int = 1
    panel: bool = True
    multi_period: bool = False
    control_group: str = "nevertreated"
    base_period: str = "varying"
