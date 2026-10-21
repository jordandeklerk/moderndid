"""Shared fixtures for core tests."""

import contextlib
import importlib.util
import shutil
import subprocess
import sys
import warnings
from unittest.mock import Mock

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
    load_favara_imbs,
    load_mpdta,
)
from moderndid.core.preprocess.builders import PreprocessDataBuilder
from moderndid.cupy import backend as cupy_backend
from moderndid.did.container import AGGTEResult
from moderndid.diddynamic.container import DynBalancingHistoryResult, DynBalancingResult
from moderndid.drdid.drdid import drdid
from moderndid.drdid.ipwdid import ipwdid
from moderndid.drdid.ordid import ordid
from moderndid.etwfe.container import EmfxResult
from tests.helpers import importorskip


@pytest.fixture(scope="session")
def mpdta():
    return load_mpdta()


@pytest.fixture
def mpdta_pandas(mpdta):
    """The county data as a pandas DataFrame."""
    return mpdta.to_pandas()


@pytest.fixture
def duckdb_connection():
    """In-memory DuckDB connection."""
    duckdb = importorskip("duckdb")
    return duckdb.connect()


@pytest.fixture
def mixed_pandas_data():
    """Pandas DataFrame of panel data that holds every column type polars must read."""
    pd = importorskip("pandas")
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "year": [2003, 2004, 2003, 2004, 2003, 2004],
            "y": [1.5, np.nan, 3.25, 4.0, 5.5, 6.0],
            "visits": pd.array([1, None, 3, 4, 5, 6], dtype="Int64"),
            "group": ["a", "b", None, "a", "b", "c"],
            "treated": [True, False, True, True, False, False],
            "cohort": pd.Categorical(["x", "y", "x", None, "y", "x"]),
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", None, "2020-01-04", "2020-01-05", "2020-01-06"]),
        }
    )


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


@pytest.fixture
def panel_with_repeated_pair():
    """Two units over two periods where the first unit's period 1 appears three times."""
    return pl.DataFrame({"id": [1, 1, 1, 2, 2], "time": [1, 1, 1, 1, 2], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def static_treatment_panel():
    """Balanced panel of three units over three periods whose treatment and covariate never change within a unit."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "treat": [1, 1, 1, 0, 0, 0, 1, 1, 1],
            "x": [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5],
        }
    )


@pytest.fixture
def panel_with_null_period():
    """Balanced panel of three units over three periods whose second unit has a null period in one row."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, None, 3, 1, 2, 3],
            "y": [10.0, 12.0, 15.0, 20.0, 22.0, 25.0, 30.0, 32.0, 35.0],
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
def aggte_simple(att_gt_analytical):
    return aggte(att_gt_analytical, type="simple")


@pytest.fixture(scope="session")
def ddd_agg_simple(ddd_mp_result):
    return agg_ddd(ddd_mp_result, type="simple")


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
def emfx_event_result():
    """ETWFE event study whose whole-number event times are stored as floats."""
    return EmfxResult(
        overall_att=0.2,
        overall_se=0.1,
        aggregation_type="event",
        event_times=np.array([0.0, 1.0, 2.0]),
        att_by_event=np.array([0.1, 0.2, 0.3]),
        se_by_event=np.array([0.05, 0.06, 0.07]),
    )


@pytest.fixture
def emfx_simple_result():
    """ETWFE simple aggregation, which holds one overall effect and no event times."""
    return EmfxResult(overall_att=0.2, overall_se=0.1, aggregation_type="simple")


@pytest.fixture
def aggte_fractional_event_times():
    """Event study whose event times fall halfway between periods."""
    return AGGTEResult(
        overall_att=0.2,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-0.5, 0.5, 1.5]),
        att_by_event=np.array([0.0, 0.2, 0.3]),
        se_by_event=np.array([0.05, 0.06, 0.07]),
    )


@pytest.fixture
def aggte_with_reference_row():
    """Event study stored as floats whose reference period at -1 has no standard error."""
    return AGGTEResult(
        overall_att=0.2,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-2.0, -1.0, 0.0]),
        att_by_event=np.array([0.05, 0.0, 0.3]),
        se_by_event=np.array([0.05, np.nan, 0.07]),
    )


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


@pytest.fixture
def dyn_balancing_panel_with_nan():
    """Panel of 40 units over three periods with a NaN period in one row and a NaN unit in another."""
    rng = np.random.default_rng(3)
    ids = np.repeat(np.arange(40.0), 3)
    t = np.tile(np.arange(1.0, 4.0), 40)
    d = np.where(ids < 20, np.tile([0.0, 1.0, 1.0], 40), 0.0)
    y = rng.normal(size=ids.size) + d
    x = rng.normal(size=ids.size)
    t[7], ids[30] = np.nan, np.nan
    return pl.DataFrame({"id": ids, "time": t, "d": d, "y": y, "x": x})


@pytest.fixture
def builder():
    return PreprocessDataBuilder()


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(42)
    n_units = 60
    n_periods = 4
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    treat_time = np.where(np.arange(n_units) < 20, 3, np.where(np.arange(n_units) < 40, 4, 0))
    groups = np.repeat(treat_time, n_periods)
    y = rng.standard_normal(n_units * n_periods) + (groups > 0).astype(float) * 2.0
    x1 = rng.standard_normal(n_units * n_periods)
    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "group": groups,
            "x1": x1,
        }
    )


@pytest.fixture
def cont_did_panel_in_years():
    """Continuous treatment panel whose periods and groups run from 2001 to 2004 instead of 1 to 4."""
    return gen_cont_did_data(n=100, num_time_periods=4, seed=42).with_columns(
        (pl.col("time_period") + 2000).alias("time_period"),
        pl.when(pl.col("G") > 0).then(pl.col("G") + 2000).otherwise(0).alias("G"),
    )


@pytest.fixture
def bootstrap_data():
    """Influence function matrix of 100 units and five columns for the multiplier bootstrap."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 5))


@pytest.fixture
def cluster_data():
    """Influence function matrix of 100 units and their cluster ids for ten clusters of ten units each."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((100, 5)), np.repeat(np.arange(10), 10)


@pytest.fixture
def rademacher_keys():
    """Random 64-bit keys of 200 units for the keyed bootstrap multipliers."""
    return np.random.default_rng(42).integers(0, 2**64, size=200, dtype=np.uint64)


@pytest.fixture
def cuda_probes(monkeypatch):
    """Report CuPy as missing and answer the nvidia-smi and nvcc probes from a mock that fails by default."""
    probes = Mock(side_effect=FileNotFoundError("nvidia-smi"))
    monkeypatch.setattr(cupy_backend, "HAS_CUPY", False)
    monkeypatch.setattr(cupy_backend, "_CUPY_IMPORT_ERROR", None)
    monkeypatch.setattr(subprocess, "check_output", probes)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/local/cuda/bin/{name}")
    cupy_backend._detect_cuda_version.cache_clear()
    yield probes
    cupy_backend._detect_cuda_version.cache_clear()


@pytest.fixture
def broken_cupy(tmp_path, monkeypatch):
    """Put a CuPy package that fails to import first on the import path and return its error message."""
    message = "libcudart.so.12: cannot open shared object file: No such file or directory"
    package = tmp_path / "cupy"
    package.mkdir()
    (package / "__init__.py").write_text(f"raise ImportError({message!r})\n")
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    return message


@pytest.fixture
def load_backend():
    """Return a loader that runs a fresh copy of the backend module and its CuPy import."""

    def load():
        spec = importlib.util.spec_from_file_location("fresh_backend", cupy_backend.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.fixture
def recorded_warnings():
    """Record the category name and message of every user warning raised in a block."""

    @contextlib.contextmanager
    def record():
        found = set()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                yield found
            finally:
                found.update(
                    (w.category.__name__, str(w.message)) for w in caught if issubclass(w.category, UserWarning)
                )

    return record


@pytest.fixture(scope="session")
def mpdta_unbalanced(mpdta):
    """mpdta without the 2005 rows of every seventh county."""
    return mpdta.filter(~((pl.col("countyreal") % 7 == 0) & (pl.col("year") == 2005)))


@pytest.fixture
def weighted_bootstrap_inputs():
    """Two-period arrays of 200 units with an intercept and one covariate laid out as each bootstrap takes them."""
    rng = np.random.default_rng(12)
    n = 200
    x = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(float)
    y0 = 1 + x + rng.normal(size=n)
    y1 = y0 + 0.5 + 0.5 * d + rng.normal(size=n)
    weights = rng.uniform(0.5, 1.5, size=n)
    covariates = np.column_stack([np.ones(n), x])
    post = (np.arange(n) % 2).astype(float)
    return {
        "panel": {"delta_y": y1 - y0, "d": d, "x": covariates, "i_weights": weights},
        "stacked_panel": {
            "y": np.column_stack([y0, y1]).ravel(),
            "d": np.repeat(d, 2),
            "post": np.tile([0.0, 1.0], n),
            "x": np.repeat(covariates, 2, axis=0),
            "i_weights": np.repeat(weights, 2),
        },
        "repeated_cross_sections": {
            "y": np.where(post == 1, y1, y0),
            "post": post,
            "d": d,
            "x": covariates,
            "i_weights": weights,
        },
    }


@pytest.fixture(scope="session")
def ddd_two_period_data():
    """Two-period triple-difference panel of 500 units with four covariates."""
    return gen_ddd_2periods(n=500, dgp_type=1, random_state=42)["data"]


@pytest.fixture(scope="session")
def iv_data():
    """Outcome that is a sine of x and an instrument w that shifts x for 500 units."""
    rng = np.random.default_rng(7)
    n = 500
    w = rng.uniform(size=n)
    x = 0.7 * w + 0.3 * rng.uniform(size=n)
    return {"y": np.sin(2 * x) + 0.2 * rng.normal(size=n), "x": x, "w": w}


@pytest.fixture(scope="session")
def cont_did_four_periods():
    """Continuous treatment panel of 200 units over four periods with pre-treatment cells."""
    return gen_cont_did_data(n=200, num_time_periods=4, seed=42)


@pytest.fixture(scope="session")
def two_period_covariate_panel():
    """Build a two-period panel of 400 units from a treatment rule and a rule for the second covariate."""

    def build(treatment, second_covariate):
        rng = np.random.default_rng(42)
        n = 400
        x = rng.normal(size=n)
        d = treatment(x, rng)
        y0 = 1.0 + 0.5 * x + rng.normal(0, 0.5, n)
        y1 = y0 + 0.3 * d + rng.normal(0, 0.5, n)
        return pl.DataFrame(
            {
                "id": np.repeat(np.arange(n), 2),
                "year": np.tile([2000, 2001], n),
                "y": np.concatenate([y0, y1]),
                "treat": np.repeat(d, 2),
                "x": np.repeat(x, 2),
                "x2": np.repeat(second_covariate(x, rng), 2),
            }
        )

    return build


@pytest.fixture(scope="session")
def collinear_covariate_panel(two_period_covariate_panel):
    """Two-period panel whose second covariate is twice the first."""
    return two_period_covariate_panel(lambda x, rng: rng.binomial(1, 0.5, x.size), lambda x, rng: 2 * x)


@pytest.fixture(scope="session")
def separated_treatment_panel(two_period_covariate_panel):
    """Two-period panel whose treated units are exactly those with the first covariate above 0.8."""
    return two_period_covariate_panel(lambda x, rng: (x > 0.8).astype(int), lambda x, rng: rng.normal(size=x.size))


@pytest.fixture
def tiny_gpu_memory_pool(cp):
    """Cap CuPy's default memory pool at one kilobyte and restore its limit afterwards."""
    pool = cp.get_default_memory_pool()
    limit = pool.get_limit()
    pool.free_all_blocks()
    pool.set_limit(size=1024)
    yield
    pool.set_limit(size=limit)
