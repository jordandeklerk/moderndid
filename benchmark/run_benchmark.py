"""CLI entry point for running benchmarks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from benchmark.config import BENCHMARK_SUITES, BenchmarkConfig
from benchmark.dgp.staggered_did import StaggeredDIDDGP
from benchmark.results.storage import BenchmarkResult, ResultStorage
from benchmark.runners.python_runner import PythonBenchmarkRunner
from benchmark.runners.r_runner import R_AVAILABLE, RBenchmarkRunner

logger = logging.getLogger(__name__)


def run_single_benchmark(
    config: BenchmarkConfig,
    python_runner: PythonBenchmarkRunner,
    r_runner: RBenchmarkRunner | None,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    logger.info(
        "Generating data: %d units, %d periods, %d groups",
        config.n_units,
        config.n_periods,
        config.n_groups,
    )

    dgp = StaggeredDIDDGP(
        n_units=config.n_units,
        n_periods=config.n_periods,
        n_groups=config.n_groups,
        n_covariates=config.n_covariates,
        random_seed=config.random_seed,
    )
    data_result = dgp.generate_data()
    data = data_result["df"]

    logger.info("Running Python benchmark (%d runs)...", config.n_runs)

    python_result = python_runner.time_att_gt(
        data=data,
        est_method=config.est_method,
        control_group=config.control_group,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        n_warmup=config.n_warmup,
        n_runs=config.n_runs,
    )

    if python_result.success:
        logger.info("Python: %.4fs (std: %.4fs)", python_result.mean_time, python_result.std_time)
    else:
        logger.error("Python: FAILED - %s", python_result.error)

    r_result = None
    if r_runner is not None:
        logger.info("Running R benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_att_gt(
            data=data,
            est_method=config.est_method,
            control_group=config.control_group,
            boot=config.boot,
            biters=config.biters,
            xformla=config.xformla,
            n_warmup=config.n_warmup,
            n_runs=config.n_runs,
        )

        if r_result.success:
            logger.info("R: %.4fs (std: %.4fs)", r_result.mean_time, r_result.std_time)
        else:
            logger.error("R: FAILED - %s", r_result.error)

    speedup = float("nan")
    if python_result.success and r_result is not None and r_result.success and python_result.mean_time > 0:
        speedup = r_result.mean_time / python_result.mean_time

    if speedup == speedup:
        faster = "Python faster" if speedup > 1 else "R faster"
        logger.info("Speedup: %.2fx (%s)", speedup, faster)

    return BenchmarkResult(
        n_units=config.n_units,
        n_periods=config.n_periods,
        n_groups=config.n_groups,
        n_covariates=config.n_covariates,
        est_method=config.est_method,
        control_group=config.control_group,
        boot=config.boot,
        biters=config.biters,
        xformla=config.xformla,
        python_mean_time=python_result.mean_time,
        python_std_time=python_result.std_time,
        python_min_time=python_result.min_time,
        python_max_time=python_result.max_time,
        python_success=python_result.success,
        python_error=python_result.error,
        r_mean_time=r_result.mean_time if r_result else float("nan"),
        r_std_time=r_result.std_time if r_result else float("nan"),
        r_min_time=r_result.min_time if r_result else float("nan"),
        r_max_time=r_result.max_time if r_result else float("nan"),
        r_success=r_result.success if r_result else False,
        r_error=r_result.error if r_result else "Skipped",
        speedup=speedup,
        n_observations=data_result["n_observations"],
        n_estimates=python_result.n_estimates,
        timestamp=datetime.now().isoformat(),
    )


def run_benchmark_suite(
    configs: list[BenchmarkConfig],
    python_only: bool = False,
) -> list[BenchmarkResult]:
    """Run a suite of benchmark configurations."""
    python_runner = PythonBenchmarkRunner()
    r_runner = None if python_only else (RBenchmarkRunner() if R_AVAILABLE else None)

    if r_runner is None and not python_only:
        logger.warning("R or did package not available, running Python-only benchmarks")

    results = []
    for i, config in enumerate(configs):
        logger.info("Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_benchmark(config, python_runner, r_runner)
        results.append(result)

    return results


def main():
    """Run benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python att_gt vs R did package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--suite",
        type=str,
        choices=list(BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite to run",
    )
    parser.add_argument("--n-units", type=int, default=1000, help="Number of units")
    parser.add_argument("--n-periods", type=int, default=5, help="Number of time periods")
    parser.add_argument("--n-groups", type=int, default=3, help="Number of treatment groups")
    parser.add_argument("--n-covariates", type=int, default=0, help="Number of covariates")
    parser.add_argument(
        "--est-method",
        type=str,
        default="dr",
        choices=["dr", "ipw", "reg"],
        help="Estimation method",
    )
    parser.add_argument("--boot", action="store_true", help="Use bootstrap")
    parser.add_argument("--biters", type=int, default=100, help="Bootstrap iterations")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--python-only", action="store_true", help="Skip R benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmark/output", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )

    if args.suite:
        configs = BENCHMARK_SUITES[args.suite]
        suite_name = args.suite
    else:
        configs = [
            BenchmarkConfig(
                n_units=args.n_units,
                n_periods=args.n_periods,
                n_groups=args.n_groups,
                n_covariates=args.n_covariates,
                est_method=args.est_method,
                boot=args.boot,
                biters=args.biters,
                n_warmup=args.warmup,
                n_runs=args.runs,
                random_seed=args.seed,
            )
        ]
        suite_name = "custom"

    logger.info("Running benchmark suite: %s", suite_name)
    logger.info("Number of configurations: %d", len(configs))
    logger.info("Python-only: %s", args.python_only)

    results = run_benchmark_suite(configs, python_only=args.python_only)

    storage = ResultStorage(output_dir=args.output_dir)
    csv_filename = storage.generate_filename(suite_name, "csv")
    json_filename = storage.generate_filename(suite_name, "json")

    csv_path = storage.save_csv(results, csv_filename)
    json_path = storage.save_json(results, json_filename)

    logger.info("Results saved to:")
    logger.info("  CSV: %s", csv_path)
    logger.info("  JSON: %s", json_path)

    logger.info("Summary:")
    for r in results:
        status = "OK" if r.python_success else "FAILED"
        msg = f"  {r.n_units} units, {r.n_periods} periods: Python {r.python_mean_time:.4f}s [{status}]"
        if r.r_success:
            msg += f", R {r.r_mean_time:.4f}s, Speedup {r.speedup:.2f}x"
        logger.info(msg)


if __name__ == "__main__":
    main()
