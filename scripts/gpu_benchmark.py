"""GPU benchmark: CPU vs CuPy performance for moderndid estimators."""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp

from benchmark.did.dgp import StaggeredDIDDGP
from moderndid import att_gt, load_mpdta, set_backend

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _bench(fn, *, n_warmup=1, n_runs=3):
    for _ in range(n_warmup):
        gc.collect()
        fn()
    times = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)), times


def _log_table(rows, header=("Config", "CPU (s)", "GPU (s)", "Speedup")):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    log.info(fmt.format(*header))
    log.info(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        log.info(fmt.format(*row))


def _make_att_gt_runner(data, *, boot=False, biters=1000):
    def _run():
        return att_gt(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="first_treat",
            xformla="~1",
            est_method="dr",
            control_group="nevertreated",
            boot=boot,
            biters=biters,
        )

    return _run


def _main():
    log.info("=" * 60)
    log.info("GPU Environment")
    log.info("=" * 60)

    log.info("CuPy version   : %s", cp.__version__)
    log.info("CUDA version    : %s", cp.cuda.runtime.runtimeGetVersion())
    try:
        dev = cp.cuda.Device(0)
        name = dev.attributes["DeviceName"]
        gpu_name = name.decode() if isinstance(name, bytes) else str(name)
    except (KeyError, AttributeError):
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    log.info("GPU device      : %s", gpu_name)

    set_backend("cupy")
    log.info("CuPy backend activated successfully")
    set_backend("numpy")

    log.info("")
    log.info("=" * 60)
    log.info("Correctness Verification")
    log.info("=" * 60)

    mpdta = load_mpdta()

    set_backend("numpy")
    cpu_result = att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
    )

    set_backend("cupy")
    gpu_result = att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        est_method="dr",
        control_group="nevertreated",
    )
    set_backend("numpy")

    max_att_diff = float(np.max(np.abs(cpu_result.att_gt - gpu_result.att_gt)))
    max_se_diff = float(np.max(np.abs(cpu_result.se_gt - gpu_result.se_gt)))

    log.info("")
    log.info("%-12s %10s %10s %10s %10s", "(g,t)", "CPU ATT", "GPU ATT", "CPU SE", "GPU SE")
    log.info("-" * 54)
    for i in range(len(cpu_result.att_gt)):
        g, t = int(cpu_result.groups[i]), int(cpu_result.times[i])
        log.info(
            "(%d,%d)  %10.6f  %10.6f  %10.6f  %10.6f",
            g,
            t,
            cpu_result.att_gt[i],
            gpu_result.att_gt[i],
            cpu_result.se_gt[i],
            gpu_result.se_gt[i],
        )

    atol = 1e-6
    assert max_att_diff < atol, f"ATT mismatch: max diff = {max_att_diff}"
    assert max_se_diff < atol, f"SE mismatch: max diff = {max_se_diff}"
    log.info("")
    log.info("Max |ATT diff| = %.2e  (tol %s)", max_att_diff, atol)
    log.info("Max |SE diff|  = %.2e  (tol %s)", max_se_diff, atol)
    log.info("PASS: CPU and GPU estimates match.")

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: att_gt (analytical SE)")
    log.info("=" * 60)

    rows = []
    for n_units in [1_000, 5_000, 10_000, 50_000, 100_000]:
        dgp = StaggeredDIDDGP(n_units=n_units, n_periods=5, n_groups=3, random_seed=42)
        data = dgp.generate_data()["df"]
        runner = _make_att_gt_runner(data)

        set_backend("numpy")
        cpu_mean, cpu_std, _ = _bench(runner)

        set_backend("cupy")
        gpu_mean, gpu_std, _ = _bench(runner)
        set_backend("numpy")

        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
        rows.append(
            (
                f"{n_units:,} units",
                f"{cpu_mean:.3f} +/- {cpu_std:.3f}",
                f"{gpu_mean:.3f} +/- {gpu_std:.3f}",
                f"{speedup:.2f}x",
            )
        )

    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: att_gt (multiplier bootstrap, biters=500)")
    log.info("=" * 60)

    rows = []
    for n_units in [1_000, 5_000, 10_000, 50_000, 100_000]:
        dgp = StaggeredDIDDGP(n_units=n_units, n_periods=5, n_groups=3, random_seed=42)
        data = dgp.generate_data()["df"]
        runner = _make_att_gt_runner(data, boot=True, biters=500)

        set_backend("numpy")
        cpu_mean, cpu_std, _ = _bench(runner)

        set_backend("cupy")
        gpu_mean, gpu_std, _ = _bench(runner)
        set_backend("numpy")

        speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
        rows.append(
            (
                f"{n_units:,} units",
                f"{cpu_mean:.3f} +/- {cpu_std:.3f}",
                f"{gpu_mean:.3f} +/- {gpu_std:.3f}",
                f"{speedup:.2f}x",
            )
        )

    log.info("")
    _log_table(rows)


if __name__ == "__main__":
    _main()
