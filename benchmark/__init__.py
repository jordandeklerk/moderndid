"""Benchmark suite for comparing Python estimators vs R packages."""

from benchmark.config import (
    ATTGT_BENCHMARK_SUITES,
    DDD_BENCHMARK_SUITES,
    ATTgtBenchmarkConfig,
    BaseBenchmarkConfig,
    DDDBenchmarkConfig,
)
from benchmark.dgp.staggered_did import StaggeredDIDDGP
from benchmark.results.storage import ATTgtBenchmarkResult, DDDBenchmarkResult, ResultStorage
from benchmark.runners.python_runner import PythonBenchmarkRunner
from benchmark.runners.r_runner import RBenchmarkRunner

__all__ = [
    "ATTGT_BENCHMARK_SUITES",
    "DDD_BENCHMARK_SUITES",
    "ATTgtBenchmarkConfig",
    "ATTgtBenchmarkResult",
    "BaseBenchmarkConfig",
    "DDDBenchmarkConfig",
    "DDDBenchmarkResult",
    "PythonBenchmarkRunner",
    "RBenchmarkRunner",
    "ResultStorage",
    "StaggeredDIDDGP",
]
