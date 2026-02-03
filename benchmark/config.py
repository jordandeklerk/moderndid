"""Benchmark configurations and predefined suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BaseBenchmarkConfig:
    """Base configuration shared by all benchmark types."""

    n_units: int = 1000
    est_method: str = "dr"
    boot: bool = False
    biters: int = 100
    xformla: str = "~1"
    n_warmup: int = 1
    n_runs: int = 5
    random_seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ATTgtBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for att_gt benchmark runs."""

    n_periods: int = 5
    n_groups: int = 3
    n_covariates: int = 0
    control_group: str = "nevertreated"


@dataclass
class DDDBenchmarkConfig(BaseBenchmarkConfig):
    """Configuration for DDD benchmark runs."""

    dgp_type: int = 1
    panel: bool = True
    multi_period: bool = False
    control_group: str = "nevertreated"
    base_period: str = "varying"
    xformla: str = "~ cov1 + cov2 + cov3 + cov4"


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

DDD_BENCHMARK_SUITES: dict[str, list[DDDBenchmarkConfig]] = {
    "scaling_units_2period": [
        DDDBenchmarkConfig(n_units=100, multi_period=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=False),
        DDDBenchmarkConfig(n_units=5000, multi_period=False),
        DDDBenchmarkConfig(n_units=10000, multi_period=False),
        DDDBenchmarkConfig(n_units=50000, multi_period=False),
        DDDBenchmarkConfig(n_units=100000, multi_period=False),
    ],
    "scaling_units_multiperiod": [
        DDDBenchmarkConfig(n_units=100, multi_period=True),
        DDDBenchmarkConfig(n_units=500, multi_period=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=True),
        DDDBenchmarkConfig(n_units=5000, multi_period=True),
        DDDBenchmarkConfig(n_units=10000, multi_period=True),
        DDDBenchmarkConfig(n_units=50000, multi_period=True),
    ],
    "est_methods": [
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="dr"),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="ipw"),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, est_method="reg"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="dr"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="ipw"),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, est_method="reg"),
    ],
    "bootstrap": [
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=100),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=500),
        DDDBenchmarkConfig(n_units=500, multi_period=False, boot=True, biters=1000),
    ],
    "panel_vs_rcs": [
        DDDBenchmarkConfig(n_units=1000, multi_period=False, panel=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=False, panel=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, panel=True),
        DDDBenchmarkConfig(n_units=1000, multi_period=True, panel=False),
    ],
    "quick": [
        DDDBenchmarkConfig(n_units=100, multi_period=False),
        DDDBenchmarkConfig(n_units=500, multi_period=False),
        DDDBenchmarkConfig(n_units=1000, multi_period=False),
    ],
}
