"""Benchmark CPU vs CuPy performance for moderndid estimators."""

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
from moderndid import (
    att_gt,
    ddd_mp,
    ddd_panel,
    drdid_panel,
    gen_dgp_mult_periods,
    load_mpdta,
    set_backend,
)

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


def _bench_loop(make_data_fn, make_runner_fn, n_units_list):
    rows = []
    for n_units in n_units_list:
        data_args = make_data_fn(n_units)
        runner = make_runner_fn(*data_args) if isinstance(data_args, tuple) else make_runner_fn(data_args)

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
    return rows


def _gen_drdid_data(n_units, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    n_treat = n_units // 2
    d = np.concatenate([np.ones(n_treat), np.zeros(n_units - n_treat)])

    X = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    beta = np.array([1.0, 0.5, -0.3])

    y0 = X @ beta + rng.standard_normal(n_units) * 0.5
    y1 = y0 + d * 1.0 + rng.standard_normal(n_units) * 0.5

    return y1, y0, d, X


def _gen_ddd_panel_data(n_units, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    n_per = n_units // 4
    sizes = [n_per, n_per, n_per, n_units - 3 * n_per]
    subgroup = np.concatenate([np.full(s, sg) for sg, s in zip([1, 2, 3, 4], sizes, strict=True)])

    X = np.column_stack([np.ones(n_units), rng.standard_normal((n_units, 2))])
    beta = np.array([1.0, 0.5, -0.3])

    y0 = X @ beta + rng.standard_normal(n_units) * 0.5
    att = 1.0
    y1 = y0 + (subgroup == 4).astype(float) * att + rng.standard_normal(n_units) * 0.5

    return y1, y0, subgroup, X


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


def _make_ddd_mp_runner(data, *, boot=False, biters=1000):
    def _run():
        return ddd_mp(
            data=data,
            y_col="y",
            time_col="time",
            id_col="id",
            group_col="group",
            partition_col="partition",
            est_method="dr",
            control_group="nevertreated",
            base_period="universal",
            boot=boot,
            biters=biters,
        )

    return _run


def _make_drdid_panel_runner(y1, y0, d, covariates, *, boot=False, nboot=999):
    def _run():
        return drdid_panel(
            y1=y1,
            y0=y0,
            d=d,
            covariates=covariates,
            boot=boot,
            boot_type="multiplier",
            nboot=nboot,
            influence_func=True,
        )

    return _run


def _make_ddd_panel_runner(y1, y0, subgroup, covariates, *, boot=False, biters=1000):
    def _run():
        return ddd_panel(
            y1=y1,
            y0=y0,
            subgroup=subgroup,
            covariates=covariates,
            est_method="dr",
            boot=boot,
            boot_type="multiplier",
            biters=biters,
            influence_func=True,
        )

    return _run


def _verify_att_gt(mpdta):
    log.info("")
    log.info("--- att_gt (multi-period DiD) ---")

    kwargs = dict(
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
    cpu = att_gt(**kwargs)

    set_backend("cupy")
    gpu = att_gt(**kwargs)
    set_backend("numpy")

    max_att = float(np.max(np.abs(cpu.att_gt - gpu.att_gt)))
    max_se = float(np.max(np.abs(cpu.se_gt - gpu.se_gt)))

    log.info("%-12s %10s %10s %10s %10s", "(g,t)", "CPU ATT", "GPU ATT", "CPU SE", "GPU SE")
    log.info("-" * 54)
    for i in range(len(cpu.att_gt)):
        g, t = int(cpu.groups[i]), int(cpu.times[i])
        log.info(
            "(%d,%d)  %10.6f  %10.6f  %10.6f  %10.6f",
            g,
            t,
            cpu.att_gt[i],
            gpu.att_gt[i],
            cpu.se_gt[i],
            gpu.se_gt[i],
        )

    atol = 1e-6
    assert max_att < atol, f"ATT mismatch: max diff = {max_att}"
    assert max_se < atol, f"SE mismatch: max diff = {max_se}"
    log.info("Max |ATT diff| = %.2e  Max |SE diff| = %.2e  (tol %s)", max_att, max_se, atol)
    log.info("PASS")


def _verify_drdid_panel():
    log.info("")
    log.info("--- drdid_panel (2-period DR DiD) ---")

    y1, y0, d, X = _gen_drdid_data(5000)

    set_backend("numpy")
    cpu = drdid_panel(y1, y0, d, X, influence_func=True)

    set_backend("cupy")
    gpu = drdid_panel(y1, y0, d, X, influence_func=True)
    set_backend("numpy")

    att_diff = abs(cpu.att - gpu.att)
    se_diff = abs(cpu.se - gpu.se)

    log.info("CPU: ATT=%.6f  SE=%.6f", cpu.att, cpu.se)
    log.info("GPU: ATT=%.6f  SE=%.6f", gpu.att, gpu.se)

    atol = 1e-6
    assert att_diff < atol, f"ATT mismatch: diff = {att_diff}"
    assert se_diff < atol, f"SE mismatch: diff = {se_diff}"
    log.info("|ATT diff| = %.2e  |SE diff| = %.2e  (tol %s)", att_diff, se_diff, atol)
    log.info("PASS")


def _verify_ddd_panel():
    log.info("")
    log.info("--- ddd_panel (2-period DDD) ---")

    y1, y0, subgroup, X = _gen_ddd_panel_data(5000)

    set_backend("numpy")
    cpu = ddd_panel(y1, y0, subgroup, X, influence_func=True)

    set_backend("cupy")
    gpu = ddd_panel(y1, y0, subgroup, X, influence_func=True)
    set_backend("numpy")

    att_diff = abs(cpu.att - gpu.att)
    se_diff = abs(cpu.se - gpu.se)

    log.info("CPU: ATT=%.6f  SE=%.6f", cpu.att, cpu.se)
    log.info("GPU: ATT=%.6f  SE=%.6f", gpu.att, gpu.se)

    atol = 1e-6
    assert att_diff < atol, f"ATT mismatch: diff = {att_diff}"
    assert se_diff < atol, f"SE mismatch: diff = {se_diff}"
    log.info("|ATT diff| = %.2e  |SE diff| = %.2e  (tol %s)", att_diff, se_diff, atol)
    log.info("PASS")


def _verify_ddd_mp():
    log.info("")
    log.info("--- ddd_mp (multi-period DDD) ---")

    dgp_result = gen_dgp_mult_periods(n=500, dgp_type=1, panel=True, random_state=42)
    data = dgp_result["data"]

    kwargs = dict(
        data=data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="dr",
        control_group="nevertreated",
        base_period="universal",
    )

    set_backend("numpy")
    cpu = ddd_mp(**kwargs)

    set_backend("cupy")
    gpu = ddd_mp(**kwargs)
    set_backend("numpy")

    max_att = float(np.max(np.abs(cpu.att - gpu.att)))
    max_se = float(np.max(np.abs(np.nan_to_num(cpu.se) - np.nan_to_num(gpu.se))))

    log.info("%-12s %10s %10s %10s %10s", "(g,t)", "CPU ATT", "GPU ATT", "CPU SE", "GPU SE")
    log.info("-" * 54)
    for i in range(len(cpu.att)):
        g, t = int(cpu.groups[i]), int(cpu.times[i])
        log.info(
            "(%d,%d)  %10.6f  %10.6f  %10.6f  %10.6f",
            g,
            t,
            cpu.att[i],
            gpu.att[i],
            cpu.se[i],
            gpu.se[i],
        )

    atol = 1e-6
    assert max_att < atol, f"ATT mismatch: max diff = {max_att}"
    assert max_se < atol, f"SE mismatch: max diff = {max_se}"
    log.info("Max |ATT diff| = %.2e  Max |SE diff| = %.2e  (tol %s)", max_att, max_se, atol)
    log.info("PASS")


SCALING_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]


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
    _verify_att_gt(mpdta)
    _verify_drdid_panel()
    _verify_ddd_panel()
    _verify_ddd_mp()

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: drdid_panel (analytical SE)")
    log.info("=" * 60)

    rng = np.random.default_rng(42)

    def _drdid_data(n):
        return _gen_drdid_data(n, rng=rng)

    def _drdid_runner(y1, y0, d, X):
        return _make_drdid_panel_runner(y1, y0, d, X)

    rows = _bench_loop(_drdid_data, _drdid_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: drdid_panel (multiplier bootstrap, biters=500)")
    log.info("=" * 60)

    rng = np.random.default_rng(42)

    def _drdid_boot_runner(y1, y0, d, X):
        return _make_drdid_panel_runner(y1, y0, d, X, boot=True, nboot=500)

    rows = _bench_loop(_drdid_data, _drdid_boot_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: ddd_panel (analytical SE)")
    log.info("=" * 60)

    rng = np.random.default_rng(42)

    def _ddd_data(n):
        return _gen_ddd_panel_data(n, rng=rng)

    def _ddd_runner(y1, y0, subgroup, X):
        return _make_ddd_panel_runner(y1, y0, subgroup, X)

    rows = _bench_loop(_ddd_data, _ddd_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: ddd_panel (multiplier bootstrap, biters=500)")
    log.info("=" * 60)

    rng = np.random.default_rng(42)

    def _ddd_boot_runner(y1, y0, subgroup, X):
        return _make_ddd_panel_runner(y1, y0, subgroup, X, boot=True, biters=500)

    rows = _bench_loop(_ddd_data, _ddd_boot_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: att_gt (analytical SE)")
    log.info("=" * 60)

    def _att_gt_data(n):
        dgp = StaggeredDIDDGP(n_units=n, n_periods=5, n_groups=3, random_seed=42)
        return dgp.generate_data()["df"]

    rows = _bench_loop(_att_gt_data, _make_att_gt_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: att_gt (multiplier bootstrap, biters=500)")
    log.info("=" * 60)

    def _att_gt_boot_runner(data):
        return _make_att_gt_runner(data, boot=True, biters=500)

    rows = _bench_loop(_att_gt_data, _att_gt_boot_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: ddd_mp (analytical SE)")
    log.info("=" * 60)

    def _ddd_mp_data(n):
        return gen_dgp_mult_periods(n=n, dgp_type=1, panel=True, random_state=42)["data"]

    rows = _bench_loop(_ddd_mp_data, _make_ddd_mp_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)

    log.info("")
    log.info("=" * 60)
    log.info("Benchmark: ddd_mp (multiplier bootstrap, biters=500)")
    log.info("=" * 60)

    def _ddd_mp_boot_runner(data):
        return _make_ddd_mp_runner(data, boot=True, biters=500)

    rows = _bench_loop(_ddd_mp_data, _ddd_mp_boot_runner, SCALING_SIZES)
    log.info("")
    _log_table(rows)


if __name__ == "__main__":
    _main()
