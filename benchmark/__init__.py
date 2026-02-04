"""Benchmark suite for comparing Python estimators vs R packages."""

from benchmark.common import BaseBenchmarkResult, BaseBenchmarkRunner, ResultStorage, TimingResult
from benchmark.did import (
    ATTGT_BENCHMARK_SUITES,
    R_DID_AVAILABLE,
    ATTgtBenchmarkConfig,
    ATTgtBenchmarkResult,
    ATTgtPythonRunner,
    ATTgtRRunner,
    StaggeredDIDDGP,
)
from benchmark.didtriple import (
    DDD_BENCHMARK_SUITES,
    R_TRIPLEDIFF_AVAILABLE,
    DDDBenchmarkConfig,
    DDDBenchmarkResult,
    DDDPythonRunner,
    DDDRRunner,
)

__all__ = [
    "ATTGT_BENCHMARK_SUITES",
    "DDD_BENCHMARK_SUITES",
    "R_DID_AVAILABLE",
    "R_TRIPLEDIFF_AVAILABLE",
    "ATTgtBenchmarkConfig",
    "ATTgtBenchmarkResult",
    "ATTgtPythonRunner",
    "ATTgtRRunner",
    "BaseBenchmarkResult",
    "BaseBenchmarkRunner",
    "DDDBenchmarkConfig",
    "DDDBenchmarkResult",
    "DDDPythonRunner",
    "DDDRRunner",
    "ResultStorage",
    "StaggeredDIDDGP",
    "TimingResult",
]
