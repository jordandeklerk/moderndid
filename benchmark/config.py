"""Benchmark configurations and predefined suites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    n_units: int = 1000
    n_periods: int = 5
    n_groups: int = 3
    n_covariates: int = 0
    est_method: str = "dr"
    control_group: str = "nevertreated"
    boot: bool = False
    biters: int = 100
    xformla: str = "~1"
    n_warmup: int = 1
    n_runs: int = 5
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "n_units": self.n_units,
            "n_periods": self.n_periods,
            "n_groups": self.n_groups,
            "n_covariates": self.n_covariates,
            "est_method": self.est_method,
            "control_group": self.control_group,
            "boot": self.boot,
            "biters": self.biters,
            "xformla": self.xformla,
            "n_warmup": self.n_warmup,
            "n_runs": self.n_runs,
            "random_seed": self.random_seed,
        }


BENCHMARK_SUITES: dict[str, list[BenchmarkConfig]] = {
    "scaling_units": [
        BenchmarkConfig(n_units=100, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=5000, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=10000, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=50000, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=100000, n_periods=5, n_groups=3),
    ],
    "scaling_periods": [
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=10, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=15, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=20, n_groups=3),
    ],
    "scaling_groups": [
        BenchmarkConfig(n_units=1000, n_periods=10, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=10, n_groups=5),
        BenchmarkConfig(n_units=1000, n_periods=10, n_groups=7),
        BenchmarkConfig(n_units=1000, n_periods=10, n_groups=10),
    ],
    "est_methods": [
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="dr"),
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="ipw"),
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="reg"),
    ],
    "bootstrap": [
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=False),
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=100),
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=500),
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=1000),
    ],
    "quick": [
        BenchmarkConfig(n_units=100, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=500, n_periods=5, n_groups=3),
        BenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
    ],
    "large_scale": [
        BenchmarkConfig(n_units=100000, n_periods=10, n_groups=5),
        BenchmarkConfig(n_units=200000, n_periods=10, n_groups=5),
        BenchmarkConfig(n_units=500000, n_periods=10, n_groups=5),
        BenchmarkConfig(n_units=1000000, n_periods=5, n_groups=5),
        BenchmarkConfig(n_units=1000000, n_periods=10, n_groups=5),
        BenchmarkConfig(n_units=2000000, n_periods=5, n_groups=5),
    ],
}
