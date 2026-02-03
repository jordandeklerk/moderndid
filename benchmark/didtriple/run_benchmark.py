"""CLI entry point for running ddd benchmarks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from benchmark.common.storage import ResultStorage
from benchmark.didtriple.config import DDD_BENCHMARK_SUITES, DDDBenchmarkConfig
from benchmark.didtriple.runners import R_TRIPLEDIFF_AVAILABLE, DDDPythonRunner, DDDRRunner
from benchmark.didtriple.storage import DDDBenchmarkResult
from moderndid import gen_dgp_2periods, gen_dgp_mult_periods

logger = logging.getLogger(__name__)


def run_single_benchmark(
    config: DDDBenchmarkConfig,
    python_runner: DDDPythonRunner,
    r_runner: DDDRRunner | None,
) -> DDDBenchmarkResult:
    """Run a single DDD benchmark configuration."""
    if config.multi_period:
        logger.info(
            "Generating multi-period DDD data: %d units, dgp_type=%d, panel=%s",
            config.n_units,
            config.dgp_type,
            config.panel,
        )
        dgp_result = gen_dgp_mult_periods(
            n=config.n_units,
            dgp_type=config.dgp_type,
            panel=config.panel,
            random_state=config.random_seed,
        )
    else:
        logger.info(
            "Generating 2-period DDD data: %d units, dgp_type=%d, panel=%s",
            config.n_units,
            config.dgp_type,
            config.panel,
        )
        dgp_result = gen_dgp_2periods(
            n=config.n_units,
            dgp_type=config.dgp_type,
            panel=config.panel,
            random_state=config.random_seed,
        )

    data = dgp_result["data"]
    n_observations = len(data)

    logger.info("Running Python DDD benchmark (%d runs)...", config.n_runs)

    python_result = python_runner.time_ddd(
        data=data,
        multi_period=config.multi_period,
        panel=config.panel,
        est_method=config.est_method,
        control_group=config.control_group,
        base_period=config.base_period,
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
    if r_runner is not None and r_runner.is_available:
        logger.info("Running R DDD benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_ddd(
            data=data,
            multi_period=config.multi_period,
            panel=config.panel,
            est_method=config.est_method,
            control_group=config.control_group,
            base_period=config.base_period,
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

    if speedup == speedup:  # Check for NaN
        faster = "Python faster" if speedup > 1 else "R faster"
        logger.info("Speedup: %.2fx (%s)", speedup, faster)

    return DDDBenchmarkResult(
        n_units=config.n_units,
        dgp_type=config.dgp_type,
        panel=config.panel,
        multi_period=config.multi_period,
        est_method=config.est_method,
        control_group=config.control_group,
        base_period=config.base_period,
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
        n_observations=n_observations,
        n_estimates=python_result.n_estimates,
        timestamp=datetime.now().isoformat(),
    )


def run_benchmark_suite(
    configs: list[DDDBenchmarkConfig],
    python_only: bool = False,
) -> list[DDDBenchmarkResult]:
    """Run a suite of DDD benchmark configurations."""
    python_runner = DDDPythonRunner()
    r_runner = None if python_only else (DDDRRunner() if R_TRIPLEDIFF_AVAILABLE else None)

    if r_runner is None and not python_only:
        logger.warning("R or triplediff package not available, running Python-only benchmarks")

    results = []
    for i, config in enumerate(configs):
        logger.info("DDD Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_benchmark(config, python_runner, r_runner)
        results.append(result)

    return results


def main():
    """Run ddd benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python ddd vs R triplediff package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--suite",
        type=str,
        choices=list(DDD_BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite to run",
    )
    parser.add_argument("--n-units", type=int, default=1000, help="Number of units")
    parser.add_argument("--dgp-type", type=int, default=1, choices=[1, 2, 3, 4], help="DGP type")
    parser.add_argument("--panel", action="store_true", default=True, help="Use panel data")
    parser.add_argument("--rcs", action="store_true", help="Use repeated cross-section data")
    parser.add_argument("--multi-period", action="store_true", help="Use multi-period DGP")
    parser.add_argument("--est-method", type=str, default="dr", choices=["dr", "ipw", "reg"], help="Estimation method")
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

    storage = ResultStorage(output_dir=args.output_dir)

    panel = not args.rcs if args.rcs else args.panel

    if args.suite:
        configs = DDD_BENCHMARK_SUITES[args.suite]
        suite_name = f"ddd_{args.suite}"
    else:
        configs = [
            DDDBenchmarkConfig(
                n_units=args.n_units,
                dgp_type=args.dgp_type,
                panel=panel,
                multi_period=args.multi_period,
                est_method=args.est_method,
                boot=args.boot,
                biters=args.biters,
                n_warmup=args.warmup,
                n_runs=args.runs,
                random_seed=args.seed,
            )
        ]
        suite_name = "ddd_custom"

    logger.info("Running DDD benchmark suite: %s", suite_name)
    logger.info("Number of configurations: %d", len(configs))
    logger.info("Python-only: %s", args.python_only)

    results = run_benchmark_suite(configs, python_only=args.python_only)

    for r in results:
        status = "OK" if r.python_success else "FAILED"
        mode = "multi-period" if r.multi_period else "2-period"
        data_type = "panel" if r.panel else "RCS"
        msg = f"  {r.n_units} units ({mode}, {data_type}): Python {r.python_mean_time:.4f}s [{status}]"
        if r.r_success:
            msg += f", R {r.r_mean_time:.4f}s, Speedup {r.speedup:.2f}x"
        logger.info(msg)

    csv_filename = storage.generate_filename(suite_name, "csv")
    json_filename = storage.generate_filename(suite_name, "json")

    csv_path = storage.save_csv(results, csv_filename)
    json_path = storage.save_json(results, json_filename)

    logger.info("Results saved to:")
    logger.info("  CSV: %s", csv_path)
    logger.info("  JSON: %s", json_path)


if __name__ == "__main__":
    main()
