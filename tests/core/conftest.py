"""Shared fixtures for core tests."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import chi2, norm

from moderndid import (
    agg_ddd,
    aggte,
    att_gt,
    cont_did,
    ddd,
    did_multiplegt,
    gen_cont_did_data,
    gen_ddd_2periods,
    gen_ddd_mult_periods,
    honest_did,
    load_favara_imbs,
    load_mpdta,
)
from moderndid.diddynamic.container import DynBalancingHistoryResult, DynBalancingResult
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid


@pytest.fixture(scope="session")
def mpdta():
    return load_mpdta()


@pytest.fixture(scope="session")
def att_gt_analytical(mpdta):
    return att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        control_group="nevertreated",
        boot=False,
        cband=False,
    )


@pytest.fixture(scope="session")
def att_gt_bootstrap(mpdta):
    return att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        control_group="nevertreated",
        boot=True,
        cband=True,
        biters=100,
        random_state=42,
    )


@pytest.fixture(
    scope="session",
    params=["simple", "dynamic", "group", "calendar"],
)
def aggte_result(request, att_gt_analytical):
    return aggte(att_gt_analytical, type=request.param)


@pytest.fixture(scope="session")
def drdid_panel_data():
    rng = np.random.default_rng(42)
    n = 200
    d = rng.binomial(1, 0.5, n)
    x = rng.normal(0, 1, n)
    y0 = 1.0 + 0.5 * x + rng.normal(0, 0.5, n)
    y1 = y0 + 0.3 * d + rng.normal(0, 0.5, n)
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(n), 2),
            "year": np.tile([2000, 2001], n),
            "y": np.concatenate([y0, y1]),
            "treat": np.repeat(d, 2),
            "x": np.repeat(x, 2),
        }
    )


@pytest.fixture(scope="session", params=[drdid, ipwdid, ordid], ids=["drdid", "ipwdid", "ordid"])
def drdid_result(request, drdid_panel_data):
    return request.param(
        data=drdid_panel_data,
        yname="y",
        tname="year",
        idname="id",
        treatname="treat",
    )


@pytest.fixture(scope="session")
def ddd_mp_data():
    return gen_ddd_mult_periods(n=300, random_state=42)["data"]


@pytest.fixture(scope="session")
def ddd_mp_result(ddd_mp_data):
    return ddd(
        data=ddd_mp_data,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        est_method="reg",
        boot=False,
    )


@pytest.fixture(scope="session")
def ddd_agg_result(ddd_mp_result):
    return agg_ddd(ddd_mp_result, type="eventstudy")


@pytest.fixture(scope="session")
def didinter_result():
    data = load_favara_imbs()
    return did_multiplegt(
        data=data,
        yname="Dl_vloans_b",
        tname="year",
        idname="county",
        dname="inter_bra",
        effects=2,
        placebo=1,
    )


@pytest.fixture(scope="session")
def cont_did_result():
    data = gen_cont_did_data(n=100, num_time_periods=4, seed=42)
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="dose",
        boot=False,
    )


@pytest.fixture
def balanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 30, 32, 35],
            "x": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
        }
    )


@pytest.fixture
def unbalanced_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
            "y": [10, 12, 15, 20, 22, 25, 32, 35, 40, 42, 45],
            "treat": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
        }
    )


@pytest.fixture
def panel_with_duplicates():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2],
            "time": [1, 1, 2, 3, 1, 2, 2],
            "y": [10.0, 11.0, 12.0, 15.0, 20.0, 22.0, 24.0],
            "cat": ["a", "b", "a", "a", "c", "c", "d"],
        }
    )


@pytest.fixture
def staggered_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "y": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
            "treat": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        }
    )


@pytest.fixture(scope="session")
def aggte_dynamic(att_gt_analytical):
    return aggte(att_gt_analytical, type="dynamic")


@pytest.fixture(scope="session")
def aggte_group(att_gt_analytical):
    return aggte(att_gt_analytical, type="group")


@pytest.fixture(scope="session")
def aggte_calendar(att_gt_analytical):
    return aggte(att_gt_analytical, type="calendar")


@pytest.fixture(scope="session")
def cont_did_event():
    data = gen_cont_did_data(n=100, num_time_periods=4, seed=42)
    return cont_did(
        data=data,
        yname="Y",
        tname="time_period",
        idname="id",
        gname="G",
        dname="D",
        target_parameter="level",
        aggregation="eventstudy",
        boot=False,
    )


@pytest.fixture(scope="session")
def honest_did_result(mpdta):
    universal = att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
        control_group="nevertreated",
        base_period="universal",
        boot=False,
        cband=False,
    )
    return honest_did(
        aggte(universal, type="dynamic"),
        event_time=0,
        sensitivity_type="relative_magnitude",
        m_bar_vec=[0.0, 1.0],
        grid_points=20,
    )


@pytest.fixture
def dyn_balancing_robust_result():
    """Dynamic balancing result with robust critical values at the 10 percent level over two periods."""
    return DynBalancingResult(
        att=0.3,
        var_att=0.04,
        mu1=8.0,
        mu2=7.7,
        var_mu1=0.02,
        var_mu2=0.022,
        robust_quantile=float(np.sqrt(chi2.ppf(0.9, 4))),
        gaussian_quantile=float(norm.ppf(0.95)),
        gammas={},
        coefficients={},
        imbalances={},
        estimation_params={"alpha": 0.1, "n_periods": 2, "robust_quantile": True},
    )


@pytest.fixture
def dyn_balancing_robust_history_result(dyn_balancing_robust_result):
    """History result over lengths one and two with robust critical values at the 10 percent level."""
    results = [
        dyn_balancing_robust_result._replace(
            robust_quantile=float(np.sqrt(chi2.ppf(0.9, 2 * length))),
            estimation_params={**dyn_balancing_robust_result.estimation_params, "n_periods": length},
        )
        for length in (1, 2)
    ]
    summary = pl.DataFrame(
        {
            "period_length": [1, 2],
            "att": [r.att for r in results],
            "var_att": [r.var_att for r in results],
            "mu1": [r.mu1 for r in results],
            "var_mu1": [r.var_mu1 for r in results],
            "mu2": [r.mu2 for r in results],
            "var_mu2": [r.var_mu2 for r in results],
            "robust_quantile": [r.robust_quantile for r in results],
            "gaussian_quantile": [r.gaussian_quantile for r in results],
        }
    )
    return DynBalancingHistoryResult(summary=summary, results=results)


@pytest.fixture
def small_never_treated_panel():
    """Balanced five-period panel with two cohorts of 20 units and 6 never-treated units."""
    rng = np.random.default_rng(0)
    cohort = np.repeat([3, 4, 0], [20, 20, 6])
    ids = np.repeat(np.arange(len(cohort)), 5)
    t = np.tile(np.arange(1, 6), len(cohort))
    g = np.repeat(cohort, 5)
    y = rng.standard_normal(len(ids)) + (t >= np.where(g > 0, g, 99))
    return pl.DataFrame({"id": ids, "t": t, "g": g, "y": y})


@pytest.fixture
def small_never_treated_cross_sections():
    """Five cross sections, each with 20 units from each of two cohorts and 3 never-treated units."""
    rng = np.random.default_rng(1)
    g = np.tile(np.repeat([3, 4, 0], [20, 20, 3]), 5)
    t = np.repeat(np.arange(1, 6), 43)
    y = rng.standard_normal(len(g)) + (t >= np.where(g > 0, g, 99))
    return pl.DataFrame({"id": np.arange(len(g)), "t": t, "g": g, "y": y})


@pytest.fixture
def two_by_two_panel():
    """Four-period panel with cohorts 3 and 4 and never-treated units coded both inf and 0."""
    return pl.DataFrame(
        {
            "id": np.repeat(np.arange(8), 4),
            "period": np.tile(np.arange(1, 5), 8),
            "G": np.repeat([3, 3, 4, 4, np.inf, np.inf, 0, 0], 4).astype(float),
            "Y": np.arange(32, dtype=float),
        }
    )


@pytest.fixture
def mpdta_with_nan(mpdta):
    """mpdta with a NaN outcome in one row and a NaN covariate in another."""
    row = pl.int_range(pl.len())
    return mpdta.with_columns(
        pl.when(row == 7).then(float("nan")).otherwise(pl.col("lemp")).alias("lemp"),
        pl.when(row == 21).then(float("nan")).otherwise(pl.col("lpop")).alias("lpop"),
    )


@pytest.fixture
def cont_did_panel_with_nan():
    """Continuous treatment panel with a NaN outcome in one row and a NaN dose in another."""
    row = pl.int_range(pl.len())
    return gen_cont_did_data(n=100, num_time_periods=4, seed=42).with_columns(
        pl.when(row == 9).then(float("nan")).otherwise(pl.col("Y")).alias("Y"),
        pl.when(row == 30).then(float("nan")).otherwise(pl.col("D")).alias("D"),
    )


@pytest.fixture
def drdid_panel_with_nan(drdid_panel_data):
    """Two-period panel with a NaN outcome in one row and a NaN covariate in another."""
    row = pl.int_range(pl.len())
    return drdid_panel_data.with_columns(
        pl.when(row == 3).then(float("nan")).otherwise(pl.col("y")).alias("y"),
        pl.when(row == 50).then(float("nan")).otherwise(pl.col("x")).alias("x"),
    )


@pytest.fixture
def ddd_panel_with_nan():
    """Two-period triple-difference panel with a NaN outcome in one row."""
    data = gen_ddd_2periods(n=200, dgp_type=1, random_state=0)["data"]
    return data.with_columns(pl.when(pl.int_range(pl.len()) == 5).then(float("nan")).otherwise(pl.col("y")).alias("y"))


@pytest.fixture
def didinter_panel_with_nan():
    """Panel of 80 groups over six periods with staggered switches and a NaN outcome, treatment, and control."""
    rng = np.random.default_rng(5)
    ids = np.repeat(np.arange(80), 6)
    t = np.tile(np.arange(1, 7), 80)
    start = np.repeat(np.array([3, 4, 0, 5])[np.arange(80) % 4], 6)
    d = ((start > 0) & (t >= start)).astype(float)
    y = np.sin(ids + t) + d + rng.normal(0, 0.1, ids.size)
    x = rng.normal(size=ids.size)
    y[13], d[20], x[31] = np.nan, np.nan, np.nan
    return pl.DataFrame({"id": ids, "t": t, "d": d, "y": y, "x": x})
