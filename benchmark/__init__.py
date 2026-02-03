"""Benchmark suite for comparing Python att_gt vs R did package."""

from benchmark.config import BENCHMARK_SUITES, BenchmarkConfig
from benchmark.dgp.staggered_did import StaggeredDIDDGP
from benchmark.results.storage import BenchmarkResult, ResultStorage
from benchmark.runners.python_runner import PythonBenchmarkRunner
from benchmark.runners.r_runner import RBenchmarkRunner

__all__ = [
    "BENCHMARK_SUITES",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PythonBenchmarkRunner",
    "RBenchmarkRunner",
    "ResultStorage",
    "StaggeredDIDDGP",
]
