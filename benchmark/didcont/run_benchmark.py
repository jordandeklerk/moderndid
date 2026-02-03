"""CLI entry point for running cont_did benchmarks."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from benchmark.common.storage import ResultStorage
from benchmark.didcont.config import CONTDID_BENCHMARK_SUITES, ContDIDBenchmarkConfig
from benchmark.didcont.runners import R_CONTDID_AVAILABLE, ContDIDPythonRunner, ContDIDRRunner
from benchmark.didcont.storage import ContDIDBenchmarkResult
from moderndid import simulate_cont_did_data

logger = logging.getLogger(__name__)


def run_single_benchmark(
    config: ContDIDBenchmarkConfig,
    python_runner: ContDIDPythonRunner,
    r_runner: ContDIDRRunner | None,
) -> ContDIDBenchmarkResult:
    """Run a single cont_did benchmark configuration."""
    logger.info(
        "Generating continuous treatment DiD data: %d units, %d periods, method=%s",
        config.n_units,
        config.n_periods,
        config.dose_est_method,
    )

    data = simulate_cont_did_data(
        n=config.n_units,
        num_time_periods=config.n_periods,
        seed=config.random_seed,
    )
    n_observations = len(data)

    logger.info("Running Python cont_did benchmark (%d runs)...", config.n_runs)

    python_result = python_runner.time_cont_did(
        data=data,
        target_parameter=config.target_parameter,
        aggregation=config.aggregation,
        dose_est_method=config.dose_est_method,
        degree=config.degree,
        num_knots=config.num_knots,
        boot=config.boot,
        biters=config.biters,
        n_warmup=config.n_warmup,
        n_runs=config.n_runs,
        random_state=config.random_seed,
    )

    if python_result.success:
        logger.info("Python: %.4fs (std: %.4fs)", python_result.mean_time, python_result.std_time)
    else:
        logger.error("Python: FAILED - %s", python_result.error)

    r_result = None
    if r_runner is not None and r_runner.is_available:
        logger.info("Running R cont_did benchmark (%d runs)...", config.n_runs)

        r_result = r_runner.time_cont_did(
            data=data,
            target_parameter=config.target_parameter,
            aggregation=config.aggregation,
            dose_est_method=config.dose_est_method,
            degree=config.degree,
            num_knots=config.num_knots,
            boot=config.boot,
            biters=config.biters,
            n_warmup=config.n_warmup,
            n_runs=config.n_runs,
            random_state=config.random_seed,
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

    return ContDIDBenchmarkResult(
        n_units=config.n_units,
        n_periods=config.n_periods,
        target_parameter=config.target_parameter,
        aggregation=config.aggregation,
        dose_est_method=config.dose_est_method,
        degree=config.degree,
        num_knots=config.num_knots,
        est_method="parametric" if config.dose_est_method == "parametric" else "cck",
        boot=config.boot,
        biters=config.biters,
        xformla="~1",
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
    configs: list[ContDIDBenchmarkConfig],
    python_only: bool = False,
) -> list[ContDIDBenchmarkResult]:
    """Run a suite of cont_did benchmark configurations."""
    python_runner = ContDIDPythonRunner()
    r_runner = None if python_only else (ContDIDRRunner() if R_CONTDID_AVAILABLE else None)

    if r_runner is None and not python_only:
        logger.warning("R or contdid package not available, running Python-only benchmarks")

    results = []
    for i, config in enumerate(configs):
        logger.info("cont_did Benchmark %d/%d:", i + 1, len(configs))
        result = run_single_benchmark(config, python_runner, r_runner)
        results.append(result)

    return results


def main():
    """Run cont_did benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Benchmark Python cont_did continuous treatment DiD estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--suite",
        type=str,
        choices=list(CONTDID_BENCHMARK_SUITES.keys()),
        help="Predefined benchmark suite to run",
    )
    parser.add_argument("--n-units", type=int, default=500, help="Number of units")
    parser.add_argument("--n-periods", type=int, default=4, help="Number of time periods")
    parser.add_argument(
        "--target-parameter",
        type=str,
        default="level",
        choices=["level", "slope"],
        help="Target parameter (level=ATT, slope=ACRT)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="dose",
        choices=["dose", "eventstudy"],
        help="Aggregation type",
    )
    parser.add_argument(
        "--dose-method",
        type=str,
        default="parametric",
        choices=["parametric", "cck"],
        help="Dose estimation method",
    )
    parser.add_argument("--degree", type=int, default=3, help="B-spline degree (parametric method)")
    parser.add_argument("--num-knots", type=int, default=0, help="Number of interior knots (parametric method)")
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

    if args.suite:
        configs = CONTDID_BENCHMARK_SUITES[args.suite]
        suite_name = f"contdid_{args.suite}"
    else:
        configs = [
            ContDIDBenchmarkConfig(
                n_units=args.n_units,
                n_periods=args.n_periods,
                target_parameter=args.target_parameter,
                aggregation=args.aggregation,
                dose_est_method=args.dose_method,
                degree=args.degree,
                num_knots=args.num_knots,
                boot=args.boot,
                biters=args.biters,
                n_warmup=args.warmup,
                n_runs=args.runs,
                random_seed=args.seed,
            )
        ]
        suite_name = "contdid_custom"

    logger.info("Running cont_did benchmark suite: %s", suite_name)
    logger.info("Number of configurations: %d", len(configs))
    logger.info("Python-only: %s", args.python_only)

    results = run_benchmark_suite(configs, python_only=args.python_only)

    for r in results:
        status = "OK" if r.python_success else "FAILED"
        msg = (
            f"  {r.n_units} units, {r.n_periods} periods "
            f"({r.dose_est_method}, {r.target_parameter}): "
            f"Python {r.python_mean_time:.4f}s [{status}]"
        )
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
