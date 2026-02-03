"""Python benchmark runner for att_gt timing."""

from __future__ import annotations

import numpy as np
import polars as pl

from benchmark.runners.base import BaseBenchmarkRunner, TimingResult
from moderndid import att_gt


class PythonBenchmarkRunner(BaseBenchmarkRunner):
    """Benchmark runner for Python att_gt implementation."""

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

        try:
            for _ in range(n_warmup):
                self.gc_collect()
                run_estimation()

            times = []
            n_estimates = 0

            for _ in range(n_runs):
                self.gc_collect()
                elapsed, result = self.time_execution(run_estimation)
                times.append(elapsed)
                n_estimates = len(result.att_gt)

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
