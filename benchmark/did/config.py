"""Benchmark configurations for att_gt estimator."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ATTgtBenchmarkConfig:
    """Configuration for att_gt benchmark runs."""

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
        return asdict(self)


ATTGT_BENCHMARK_SUITES: dict[str, list[ATTgtBenchmarkConfig]] = {
    "scaling_units": [
        ATTgtBenchmarkConfig(n_units=100, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=5000, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=10000, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=50000, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=100000, n_periods=5, n_groups=3),
    ],
    "scaling_periods": [
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=10, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=15, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=20, n_groups=3),
    ],
    "scaling_groups": [
        ATTgtBenchmarkConfig(n_units=1000, n_periods=10, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=10, n_groups=5),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=10, n_groups=7),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=10, n_groups=10),
    ],
    "est_methods": [
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="dr"),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="ipw"),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3, est_method="reg"),
    ],
    "bootstrap": [
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=False),
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=100),
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=500),
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3, boot=True, biters=1000),
    ],
    "quick": [
        ATTgtBenchmarkConfig(n_units=100, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=500, n_periods=5, n_groups=3),
        ATTgtBenchmarkConfig(n_units=1000, n_periods=5, n_groups=3),
    ],
    "large_scale": [
        ATTgtBenchmarkConfig(n_units=100000, n_periods=10, n_groups=5),
        ATTgtBenchmarkConfig(n_units=200000, n_periods=10, n_groups=5),
        ATTgtBenchmarkConfig(n_units=500000, n_periods=10, n_groups=5),
        ATTgtBenchmarkConfig(n_units=1000000, n_periods=5, n_groups=5),
        ATTgtBenchmarkConfig(n_units=1000000, n_periods=10, n_groups=5),
        ATTgtBenchmarkConfig(n_units=2000000, n_periods=5, n_groups=5),
    ],
}
