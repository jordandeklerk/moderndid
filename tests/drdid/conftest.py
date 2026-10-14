"""Shared fixtures for drdid tests."""

import importlib
import types
from collections import namedtuple

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import gen_did_scalable, load_nsw
from moderndid.drdid.bootstrap import boot_mult


@pytest.fixture(scope="module")
def nsw_data():
    return load_nsw()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def dr_panel_result():
    DRDIDPanel = namedtuple("DRDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return DRDIDPanel(att=1.5, se=0.3, lci=0.9, uci=2.1, args={})


@pytest.fixture
def dr_rc_result():
    DRDIDRc = namedtuple("DRDIDRcResult", ["att", "se", "lci", "uci", "args"])
    return DRDIDRc(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def ipw_result():
    IPW = namedtuple("IPWDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return IPW(att=1.0, se=0.3, lci=0.4, uci=1.6, args={})


@pytest.fixture
def reg_result():
    Reg = namedtuple("RegDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return Reg(att=1.0, se=0.3, lci=0.4, uci=1.6, args={})


@pytest.fixture
def twfe_result():
    TWFE = namedtuple("TWFEDIDPanelResult", ["att", "se", "lci", "uci", "args"])
    return TWFE(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def unknown_result():
    Unknown = namedtuple("FooBarResult", ["att", "se", "lci", "uci", "args"])
    return Unknown(att=1.0, se=0.2, lci=0.6, uci=1.4, args={})


@pytest.fixture
def result_with_call_params():
    WithCP = namedtuple("DRDIDPanelCallResult", ["att", "se", "lci", "uci", "args", "call_params"])
    return WithCP(att=1.0, se=0.2, lci=0.6, uci=1.4, args={}, call_params={"data_shape": (500, 8)})


@pytest.fixture(params=["repeated_row", "hidden_gap", "relabeled_row"])
def nsw_duplicated(request, nsw_data):
    """NSW panel in which unit 15995 has two rows in 1975."""
    unit = pl.col("id") == 15995
    row = nsw_data.filter(unit & (pl.col("year") == 1975))
    if request.param == "repeated_row":
        return pl.concat([nsw_data, row])
    if request.param == "hidden_gap":
        return pl.concat([nsw_data.filter(~(unit & (pl.col("year") == 1978))), row])
    return pl.concat([nsw_data, row.with_columns(pl.col("re") + 1000)])


@pytest.fixture
def nsw_one_infinite(request, nsw_data):
    """NSW panel with weights in w and an infinite value in one 1978 row of unit 15995."""
    column, value = request.param
    row = (pl.col("id") == 15995) & (pl.col("year") == 1978)
    data = nsw_data.with_columns(pl.lit(1.5).alias("w"))
    return data.with_columns(pl.when(row).then(value).otherwise(pl.col(column).cast(pl.Float64)).alias(column))


@pytest.fixture
def unit_bootstrap_weights(monkeypatch):
    """Exponential bootstrap draws that all equal one and leave every weight unchanged."""

    class UnitWeights:
        def __init__(self, seed=None):
            pass

        def exponential(self, scale=1.0, size=None):
            return np.ones(size)

    monkeypatch.setattr(np.random, "RandomState", UnitWeights)


@pytest.fixture
def numpy_as_gpu_backend(monkeypatch):
    """Send the multiplier bootstrap down its GPU branch with numpy standing in for CuPy."""

    class NumpyAsGpu:
        def __getattr__(self, name):
            return getattr(np, name)

    monkeypatch.setattr(boot_mult, "get_backend", NumpyAsGpu)


@pytest.fixture(scope="module")
def trimmed_two_period_data():
    """Two-period draw in which several comparison units have propensity scores above 0.9."""
    rng = np.random.default_rng(11)
    n = 2000
    x1 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-2 * x1))).astype(float)
    y0 = 1 + x1 + rng.normal(size=n)
    y1 = y0 + 1 + 0.5 * x1**2 + rng.normal(size=n)
    post = (rng.uniform(size=n) < 0.5).astype(float)
    return {
        "y1": y1,
        "y0": y0,
        "y": np.where(post == 1, y1, y0),
        "post": post,
        "d": d,
        "x": np.column_stack([np.ones(n), x1]),
    }


@pytest.fixture(scope="module")
def rc_correct_models():
    """Repeated cross-sections in which the logistic score and linear outcome models are all correct."""
    rng = np.random.default_rng(7)
    n = 1000
    x1, x2, x3, x4 = rng.normal(size=(4, n))
    z1 = (np.exp(x1 / 2) - np.exp(0.125)) / np.sqrt((np.exp(0.25) - 1) * np.exp(0.25))
    z2 = x2 / (1 + np.exp(x1)) / 0.54164
    z3 = ((x1 * x3 / 25 + 0.6) ** 3 - 0.21887) / 0.04453
    z4 = ((x1 + x4 + 20) ** 2 - 402) / 56.63891
    score = 1 / (1 + np.exp(-0.75 * (-z1 + 0.5 * z2 - 0.25 * z3 - 0.1 * z4)))
    d = (rng.uniform(size=n) <= score).astype(int)
    index = 210 + 27.4 * z1 + 13.7 * (z2 + z3 + z4)
    v = rng.normal(loc=d * index, scale=1, size=n)
    y0 = index + v + rng.normal(size=n)
    y1 = 2 * index + v + rng.normal(size=n)
    post = (rng.uniform(size=n) <= 0.5).astype(int)
    return pl.DataFrame(
        {"year": 2000 + post, "y": np.where(post == 1, y1, y0), "d": d, "z1": z1, "z2": z2, "z3": z3, "z4": z4}
    )


@pytest.fixture(scope="module")
def rc_correct_models_arrays(rc_correct_models):
    """The rc_correct_models data as the arrays the repeated cross-section estimators take."""
    covariates = rc_correct_models.select("z1", "z2", "z3", "z4").to_numpy()
    return {
        "y": rc_correct_models["y"].to_numpy(),
        "post": (rc_correct_models["year"] == 2001).cast(pl.Int64).to_numpy(),
        "d": rc_correct_models["d"].to_numpy(),
        "covariates": np.column_stack([np.ones(rc_correct_models.height), covariates]),
    }


@pytest.fixture(scope="module")
def rc_misspecified_score_arrays():
    """Two-period repeated cross-section arrays whose logistic propensity score model is misspecified."""
    simulated = gen_did_scalable(
        n=1000, dgp_type=2, n_periods=2, n_cohorts=1, n_covariates=4, panel=False, random_state=42
    )
    data = simulated["data"]
    return {
        "y": data["y"].to_numpy(),
        "post": (data["time"] == 2).cast(pl.Int64).to_numpy(),
        "d": (data["group"] == 2).cast(pl.Int64).to_numpy(),
        "covariates": np.column_stack([np.ones(data.height), data.select("cov1", "cov2", "cov3", "cov4").to_numpy()]),
    }


@pytest.fixture(scope="module")
def rc_married_in_treated_pre():
    """Repeated cross-sections in which married equals 1 for every treated row of the pre-period."""
    rng = np.random.default_rng(6)
    n = 4000
    post = (rng.uniform(size=n) < 0.5).astype(int)
    x0 = rng.normal(size=n)
    married = (rng.uniform(size=n) < 0.5).astype(int)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x0))).astype(int)
    married[(d == 1) & (post == 0)] = 1
    y = 1 + x0 + 0.5 * married + d * post + rng.normal(size=n)
    return pl.DataFrame({"year": 2000 + post, "y": y, "d": d, "x0": x0, "married": married})


@pytest.fixture(scope="module")
def rc_binary_one_treated_pre_arrays():
    """Repeated cross-section arrays with a binary covariate and one treated row in the pre-period."""
    rng = np.random.default_rng(9)
    post = np.repeat([0, 0, 1, 1], [1, 150, 60, 150])
    d = np.repeat([1, 0, 1, 0], [1, 150, 60, 150])
    b = (rng.uniform(size=post.size) < 0.5).astype(float)
    y = 1 + b + post * (0.5 + b) + d * post + rng.normal(size=post.size)
    return {"y": y, "post": post, "d": d, "covariates": np.column_stack([np.ones(post.size), b])}


@pytest.fixture(scope="module")
def rc_binary_constant_treated_pre_arrays():
    """Repeated cross-section arrays whose treated rows in the pre-period all have a binary covariate of 1."""
    rng = np.random.default_rng(10)
    post = np.repeat([0, 0, 1, 1], [40, 150, 60, 150])
    d = np.repeat([1, 0, 1, 0], [40, 150, 60, 150])
    b = (rng.uniform(size=post.size) < 0.5).astype(float)
    b[(d == 1) & (post == 0)] = 1.0
    y = 1 + b + post * (0.5 + b) + d * post + rng.normal(size=post.size)
    return {"y": y, "post": post, "d": d, "covariates": np.column_stack([np.ones(post.size), b])}


@pytest.fixture(scope="module")
def panel_married_in_comparison():
    """Panel arrays in which married equals 1 for every comparison unit."""
    rng = np.random.default_rng(24)
    n = 2000
    x0 = rng.normal(size=n)
    married = (rng.uniform(size=n) < 0.5).astype(int)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x0))).astype(int)
    married[d == 0] = 1
    y0 = 1 + x0 + 0.5 * married + rng.normal(size=n)
    y1 = y0 + 1 + d + rng.normal(size=n)
    return {"y1": y1, "y0": y0, "d": d, "covariates": np.column_stack([np.ones(n), x0, married])}


@pytest.fixture(scope="module")
def rc_small_treated_pre():
    """Repeated cross-sections with five covariates whose treated group has 4 rows in the pre-period."""
    rng = np.random.default_rng(3)
    post = np.repeat([0, 0, 1, 1], [4, 140, 60, 140])
    d = np.repeat([1, 0, 1, 0], [4, 140, 60, 140])
    x = rng.normal(size=(post.size, 5))
    y = x @ np.arange(1, 6) + d * post + rng.normal(size=post.size)
    return pl.DataFrame({"year": 2000 + post, "y": y, "d": d, **{f"x{j}": x[:, j] for j in range(5)}})


@pytest.fixture(scope="module")
def rc_small_treated_pre_arrays(rc_small_treated_pre):
    """The rc_small_treated_pre data as the arrays the repeated cross-section estimators take."""
    data = rc_small_treated_pre
    return {
        "y": data["y"].to_numpy(),
        "post": (data["year"] == 2001).cast(pl.Int64).to_numpy(),
        "d": data["d"].to_numpy(),
        "covariates": np.column_stack([np.ones(data.height), data.select("x0", "x1", "x2", "x3", "x4").to_numpy()]),
    }


@pytest.fixture(scope="module")
def rc_small_comparison_post_arrays():
    """Repeated cross-section arrays with five covariates whose comparison group has 4 rows in the post-period."""
    rng = np.random.default_rng(4)
    post = np.repeat([0, 0, 1, 1], [60, 140, 60, 4])
    d = np.repeat([1, 0, 1, 0], [60, 140, 60, 4])
    x = rng.normal(size=(post.size, 5))
    y = x @ np.arange(1, 6) + d * post + rng.normal(size=post.size)
    return {"y": y, "post": post, "d": d, "covariates": np.column_stack([np.ones(post.size), x])}


@pytest.fixture(scope="module")
def panel_small_comparison_arrays():
    """Panel arrays with five covariates and 4 comparison units."""
    rng = np.random.default_rng(5)
    d = np.repeat([1, 0], [150, 4])
    x = rng.normal(size=(d.size, 5))
    y0 = x @ np.arange(1, 6) + rng.normal(size=d.size)
    y1 = y0 + d + rng.normal(size=d.size)
    return {"y1": y1, "y0": y0, "d": d, "covariates": np.column_stack([np.ones(d.size), x])}


@pytest.fixture(scope="module")
def panel_small_comparison(panel_small_comparison_arrays):
    """The panel_small_comparison_arrays data in the long format that drdid takes."""
    arrays = panel_small_comparison_arrays
    n = arrays["d"].size
    x = arrays["covariates"][:, 1:]
    return pl.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "year": np.repeat([2000, 2001], n),
            "y": np.r_[arrays["y0"], arrays["y1"]],
            "d": np.tile(arrays["d"], 2),
            **{f"x{j}": np.tile(x[:, j], 2) for j in range(5)},
        }
    )


@pytest.fixture(scope="module")
def two_covariate_design():
    """Repeated cross-section and panel outcomes that share a treatment indicator and two covariates."""
    rng = np.random.default_rng(8)
    n = 2000
    post = (rng.uniform(size=n) < 0.5).astype(int)
    z = rng.normal(size=(n, 2))
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * z[:, 0]))).astype(int)
    y0 = z @ np.array([1.0, -1.0]) + rng.normal(size=n)
    y1 = y0 + 0.5 + d + 0.5 * z[:, 1] + rng.normal(size=n)
    return {"post": post, "d": d, "y": np.where(post == 1, y1, y0), "y1": y1, "y0": y0, "z": z}


@pytest.fixture
def array_module_without_cond(monkeypatch):
    """Make drdid's helpers use NumPy without linalg.cond like CuPy 13."""
    linalg = types.SimpleNamespace(**{name: getattr(np.linalg, name) for name in dir(np.linalg) if name != "cond"})
    module = types.SimpleNamespace(**{name: getattr(np, name) for name in dir(np) if not name.startswith("__")})
    module.linalg = linalg
    monkeypatch.setattr(importlib.import_module("moderndid.drdid.utils"), "_array_module", lambda *arrays: module)
    return module


@pytest.fixture(scope="module")
def two_period_specs(two_covariate_design):
    """Arguments of the two-period estimators for the panel and repeated cross-section layouts of one design."""
    n = len(two_covariate_design["d"])
    shared = {
        "d": two_covariate_design["d"],
        "covariates": np.column_stack([np.ones(n), two_covariate_design["z"]]),
        "i_weights": np.random.default_rng(2).uniform(0.5, 1.5, size=n),
    }
    return {
        "panel": {"y1": two_covariate_design["y1"], "y0": two_covariate_design["y0"], **shared},
        "repeated_cross_sections": {"y": two_covariate_design["y"], "post": two_covariate_design["post"], **shared},
    }


@pytest.fixture(scope="module")
def regression_specs(two_period_specs):
    """Arguments of the four comparison-group regressions keyed by function name."""
    panel = two_period_specs["panel"]
    cross_sections = two_period_specs["repeated_cross_sections"]
    ps = 1 / (1 + np.exp(-0.5 * panel["covariates"][:, 1]))
    shared = {"d": panel["d"], "x": panel["covariates"], "i_weights": panel["i_weights"]}
    panel_spec = {"delta_y": panel["y1"] - panel["y0"], **shared}
    cross_section_spec = {
        "y": cross_sections["y"],
        "post": cross_sections["post"],
        "pre": True,
        "treat": False,
        **shared,
    }
    return {
        "ols_panel": panel_spec,
        "wols_panel": {**panel_spec, "ps": ps},
        "ols_rc": cross_section_spec,
        "wols_rc": {**cross_section_spec, "ps": ps},
    }
