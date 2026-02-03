"""Python benchmark runner for att_gt and ddd timing."""

from __future__ import annotations

import numpy as np
import polars as pl

from benchmark.runners.base import BaseBenchmarkRunner, TimingResult
from moderndid import att_gt, ddd


class PythonBenchmarkRunner(BaseBenchmarkRunner):
    """Benchmark runner for Python estimator implementations."""

    def time_att_gt(
        self,
        data: pl.DataFrame,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~1",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time Python att_gt estimation."""

        def run_estimation():
            return att_gt(
                data=data,
                yname="y",
                tname="time",
                idname="id",
                gname="first_treat",
                xformla=xformla,
                est_method=est_method,
                control_group=control_group,
                boot=boot,
                biters=biters if boot else 0,
            )

        return self._run_timed_benchmark(
            run_estimation,
            n_warmup=n_warmup,
            n_runs=n_runs,
            get_n_estimates=lambda r: len(r.att_gt),
        )

    def time_ddd(
        self,
        data: pl.DataFrame,
        multi_period: bool = False,
        panel: bool = True,
        est_method: str = "dr",
        control_group: str = "nevertreated",
        base_period: str = "varying",
        boot: bool = False,
        biters: int = 100,
        xformla: str = "~ cov1 + cov2 + cov3 + cov4",
        n_warmup: int = 1,
        n_runs: int = 5,
    ) -> TimingResult:
        """Time Python DDD estimation."""
        gname = "group" if multi_period else "state"

        def run_estimation():
            return ddd(
                data=data,
                yname="y",
                tname="time",
                idname="id",
                gname=gname,
                pname="partition",
                xformla=xformla,
                est_method=est_method,
                control_group=control_group,
                base_period=base_period,
                panel=panel,
                boot=boot,
                biters=biters if boot else 0,
            )

        def get_n_estimates(result):
            if hasattr(result, "att") and hasattr(result.att, "__len__"):
                return len(result.att)
            return 1

        return self._run_timed_benchmark(
            run_estimation,
            n_warmup=n_warmup,
            n_runs=n_runs,
            get_n_estimates=get_n_estimates,
        )

    def _run_timed_benchmark(
        self,
        run_fn,
        n_warmup: int,
        n_runs: int,
        get_n_estimates,
    ) -> TimingResult:
        """Run a timed benchmark with warmup and multiple runs."""
        try:
            for _ in range(n_warmup):
                self.gc_collect()
                run_fn()

            times = []
            n_estimates = 0

            for _ in range(n_runs):
                self.gc_collect()
                elapsed, result = self.time_execution(run_fn)
                times.append(elapsed)
                n_estimates = get_n_estimates(result)

            return TimingResult(
                mean_time=float(np.mean(times)),
                std_time=float(np.std(times)),
                min_time=float(np.min(times)),
                max_time=float(np.max(times)),
                times=times,
                n_estimates=n_estimates,
                success=True,
            )

        except (ValueError, RuntimeError, KeyError) as e:
            return TimingResult(
                mean_time=float("nan"),
                std_time=float("nan"),
                min_time=float("nan"),
                max_time=float("nan"),
                times=[],
                n_estimates=0,
                success=False,
                error=str(e),
            )
