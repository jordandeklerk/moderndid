"""Shared fixtures for drdid tests."""

from collections import namedtuple

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import gen_did_scalable, load_nsw


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
def wide_two_period_data():
    """Two-period panel from gen_did_scalable with an intercept and twenty covariates."""
    simulated = gen_did_scalable(
        n=1500, dgp_type=1, n_periods=2, n_cohorts=1, n_covariates=20, panel=True, random_state=42
    )
    data = simulated["data"].sort(["id", "time"])
    pre = data.filter(pl.col("time") == 1)
    post = data.filter(pl.col("time") == 2)
    return {
        "y1": post["y"].to_numpy(),
        "y0": pre["y"].to_numpy(),
        "d": (pre["group"] == 2).cast(pl.Int64).to_numpy(),
        "x": np.column_stack([np.ones(pre.height), pre.select([f"cov{j}" for j in range(1, 21)]).to_numpy()]),
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


@pytest.fixture(scope="module")
def nsw_changes(nsw_data):
    """NSW units sorted by id with their change in earnings from 1975 to 1978 in dre."""
    pre = nsw_data.filter(pl.col("year") == 1975).sort("id")
    post = nsw_data.filter(pl.col("year") == 1978).sort("id")
    return post.with_columns((pl.col("re") - pre["re"]).alias("dre"))


@pytest.fixture(scope="module")
def nsw_thinned(nsw_data):
    """NSW rows without the 1975 rows of the treated black units with even ids."""
    dropped = (
        (pl.col("experimental") == 1) & (pl.col("black") == 1) & (pl.col("year") == 1975) & (pl.col("id") % 2 == 0)
    )
    return nsw_data.filter(~dropped)


@pytest.fixture(scope="module")
def trimmed_two_period_frames(trimmed_two_period_data):
    """The trimmed_two_period_data draw as the long panel and repeated cross-section data that drdid takes."""
    data = trimmed_two_period_data
    n = data["d"].size
    d = data["d"].astype(int)
    panel = pl.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "year": np.repeat([2000, 2001], n),
            "y": np.r_[data["y0"], data["y1"]],
            "d": np.tile(d, 2),
            "x1": np.tile(data["x"][:, 1], 2),
        }
    )
    cross_sections = pl.DataFrame(
        {"year": 2000 + data["post"].astype(int), "y": data["y"], "d": d, "x1": data["x"][:, 1]}
    )
    return {"panel": panel, "repeated_cross_sections": cross_sections}


@pytest.fixture
def panel_matching_did():
    """Function that computes the exact matching DiD of a panel and its influence function by hand."""

    def estimate(delta_y, d, cells, weights):
        weights = weights / weights.mean()
        labels = np.unique(cells, return_inverse=True)[1]
        comparison = np.bincount(labels, weights=weights * (1 - d))
        treated = np.bincount(labels, weights=weights * d)
        mu = (np.bincount(labels, weights=weights * (1 - d) * delta_y) / comparison)[labels]
        odds = (treated / comparison)[labels]
        att = np.sum(weights * d * (delta_y - mu)) / np.sum(weights * d)
        inf_func = weights * (d * (delta_y - mu - att) - (1 - d) * odds * (delta_y - mu)) / np.mean(weights * d)
        return att, inf_func

    return estimate


@pytest.fixture
def rc_matching_did():
    """Function that averages the cell DiDs of repeated cross-sections over all treated rows by hand."""

    def estimate(y, post, d, cells, weights):
        weights = weights / weights.mean()
        d = d.astype(int)
        post = post.astype(int)
        labels = np.unique(cells, return_inverse=True)[1]
        n_cells = labels.max() + 1
        group = 4 * labels + 2 * d + post
        mass = np.bincount(group, weights=weights, minlength=4 * n_cells)
        means = np.bincount(group, weights=weights * y, minlength=4 * n_cells) / mass
        cell_means = means.reshape(n_cells, 2, 2)
        did = cell_means[:, 1, 1] - cell_means[:, 1, 0] - cell_means[:, 0, 1] + cell_means[:, 0, 0]
        share = np.bincount(labels, weights=weights * d, minlength=n_cells) / np.sum(weights * d)
        att = np.sum(share * did)
        sign = np.where(d == post, 1.0, -1.0)
        inf_func = weights * (
            d * (did[labels] - att) / np.mean(weights * d)
            + share[labels] * sign * (y - means[group]) * y.size / mass[group]
        )
        return att, inf_func

    return estimate


@pytest.fixture
def rc_period_matching_did():
    """Function that matches the treated rows of each period to the comparison cells of that period by hand."""

    def estimate(y, post, d, cells, weights):
        weights = weights / weights.mean()
        d = d.astype(int)
        post = post.astype(int)
        labels = np.unique(cells, return_inverse=True)[1]
        group = 4 * labels + 2 * d + post
        share = np.bincount(group, weights=weights, minlength=4 * (labels.max() + 1)) / y.size
        means = np.bincount(group, weights=weights * y, minlength=4 * (labels.max() + 1)) / (share * y.size)
        residual = y - means[4 * labels + post]
        treated_post = np.mean(weights * d * post)
        treated_pre = np.mean(weights * d * (1 - post))
        after = np.mean(weights * d * post * residual) / treated_post
        before = np.mean(weights * d * (1 - post) * residual) / treated_pre
        treated_share = np.where(post == 1, share[4 * labels + 3] / treated_post, share[4 * labels + 2] / treated_pre)
        inf_func = weights * (
            d * post * (residual - after) / treated_post
            - d * (1 - post) * (residual - before) / treated_pre
            - (1 - d) * (2 * post - 1) * residual * treated_share / share[4 * labels + post]
        )
        return after - before, inf_func

    return estimate


@pytest.fixture
def seeded_exponential_draws(monkeypatch):
    """Weighted bootstrap draws taken from default_rng(8) in place of the unseeded RandomState."""

    class SeededWeights:
        def __init__(self, seed=None):
            self._rng = np.random.default_rng(8)

        def exponential(self, scale=1.0, size=None):
            return self._rng.exponential(scale, size)

    monkeypatch.setattr(np.random, "RandomState", SeededWeights)


@pytest.fixture
def seeded_default_rng(monkeypatch):
    """np.random.default_rng that starts every unseeded generator from seed 6."""
    default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: default_rng(6 if seed is None else seed))


@pytest.fixture(scope="module")
def nsw_weighted(nsw_data):
    """NSW panel with exponential sampling weights in w that stay constant within each unit."""
    ids = nsw_data["id"].unique().sort()
    weights = pl.DataFrame({"id": ids, "w": np.random.default_rng(42).exponential(1, ids.len())})
    return nsw_data.join(weights, on="id", how="left", maintain_order="left")


@pytest.fixture(scope="module")
def nsw_units(nsw_weighted):
    """One row per NSW unit sorted by id with its weight in w and its change in earnings from 1975 to 1978 in dy."""
    pre = nsw_weighted.filter(pl.col("year") == 1975).sort("id")
    post = nsw_weighted.filter(pl.col("year") == 1978).sort("id")
    return post.with_columns((post["re"] - pre["re"]).alias("dy"))


@pytest.fixture(scope="module")
def trimmed_two_period_panel(trimmed_two_period_data):
    """The trimmed_two_period_data draw in the long panel format that ipwdid takes."""
    data = trimmed_two_period_data
    n = data["d"].size
    return pl.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "year": np.repeat([2000, 2001], n),
            "y": np.r_[data["y0"], data["y1"]],
            "d": np.tile(data["d"], 2),
            "x1": np.tile(data["x"][:, 1], 2),
        }
    )


@pytest.fixture(scope="module")
def trimmed_two_period_cross_sections(trimmed_two_period_data):
    """The trimmed_two_period_data draw as the repeated cross-sections that ipwdid takes."""
    data = trimmed_two_period_data
    return pl.DataFrame({"year": 2000 + data["post"], "y": data["y"], "d": data["d"], "x1": data["x"][:, 1]})


@pytest.fixture
def unit_generator_bootstrap_weights(monkeypatch):
    """Exponential draws of np.random.default_rng that all equal one and leave every weight unchanged."""

    class UnitGenerator(np.random.Generator):
        def __init__(self):
            super().__init__(np.random.PCG64(0))

        def exponential(self, scale=1.0, size=None):
            return np.ones(size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: UnitGenerator())


@pytest.fixture
def least_squares_fit():
    """Fit weighted least squares on the chosen rows by lstsq and return the fitted values of every row."""

    def fit(y, x, weights, rows):
        root = np.sqrt(weights[rows])
        coefficients = np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]
        return x @ coefficients

    return fit


@pytest.fixture
def maximum_likelihood_scores():
    """Fit a weighted logistic propensity score by Newton steps on its log likelihood and return the scores."""

    def fit(d, x, weights):
        gamma = np.zeros(x.shape[1])
        for _ in range(30):
            score = 1 / (1 + np.exp(-x @ gamma))
            hessian = (x * (weights * score * (1 - score))[:, None]).T @ x
            gamma = gamma + np.linalg.solve(hessian, x.T @ (weights * (d - score)))
        return 1 / (1 + np.exp(-x @ gamma))

    return fit


@pytest.fixture
def tilting_scores():
    """Solve the tilting conditions that match the weighted covariate means of both groups and return the scores."""

    def fit(d, x, weights):
        gamma = np.zeros(x.shape[1])
        for _ in range(30):
            tilt = weights * (1 - d) * np.exp(x @ gamma)
            gamma = gamma + np.linalg.solve((x * tilt[:, None]).T @ x, x.T @ (weights * d - tilt))
        return 1 / (1 + np.exp(-x @ gamma))

    return fit


@pytest.fixture
def panel_dr_att(least_squares_fit):
    """Compute the panel doubly robust ATT of the paper from given propensity scores."""

    def att(delta_y, d, x, weights, ps, improved, trim_level=1.0):
        odds = ps / (1 - ps)
        residual = delta_y - least_squares_fit(delta_y, x, weights * odds if improved else weights, d == 0)
        comparison = weights * odds * (1 - d) * (ps < trim_level)
        scale = np.sum(weights * d) if improved else np.sum(comparison)
        return np.average(residual, weights=weights * d) - np.sum(comparison * residual) / scale

    return att


@pytest.fixture
def rc_dr_att(least_squares_fit):
    """Compute the repeated cross-section doubly robust ATT of the paper from given propensity scores."""

    def att(y, post, d, x, weights, ps, improved, locally_efficient, trim_level=1.0):
        odds = ps / (1 - ps)
        regression_weights = weights * odds if improved else weights
        fits = [least_squares_fit(y, x, regression_weights, (d == 0) & (post == period)) for period in (0, 1)]
        residual = y - np.where(post == 1, fits[1], fits[0])
        comparison = odds * (1 - d) * (ps < trim_level)
        estimate = (
            np.average(residual, weights=weights * d * post)
            - np.average(residual, weights=weights * d * (1 - post))
            - np.average(residual, weights=weights * comparison * post)
            + np.average(residual, weights=weights * comparison * (1 - post))
        )
        if not locally_efficient:
            return estimate
        for period, sign in ((1, 1), (0, -1)):
            gap = least_squares_fit(y, x, weights, (d == 1) & (post == period)) - fits[period]
            in_period = d * (post == period)
            estimate += sign * (np.average(gap, weights=weights * d) - np.average(gap, weights=weights * in_period))
        return estimate

    return att


@pytest.fixture(scope="module")
def rc_period_balanced_binary_arrays():
    """Repeated cross-section arrays whose groups have equal weighted binary covariate shares in both periods."""
    rng = np.random.default_rng(13)
    counts = [30, 20, 60, 40, 50, 50, 70, 70]
    d = np.repeat([1, 1, 1, 1, 0, 0, 0, 0], counts)
    post = np.repeat([0, 0, 1, 1, 0, 0, 1, 1], counts)
    b = np.repeat([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], counts)
    y = 1 + b + post * (0.5 + b) + d * (0.5 + post * (1 + b)) + rng.normal(size=d.size)
    return {
        "y": y,
        "post": post,
        "d": d,
        "covariates": np.column_stack([np.ones(d.size), b]),
        "i_weights": (1 + b) * (1.5 - 0.5 * d),
    }


@pytest.fixture(scope="module")
def panel_planted_effect_arrays():
    """Panel arrays with correct score and outcome models whose treated units gain exactly one."""
    rng = np.random.default_rng(42)
    n = 5000
    x = rng.normal(size=(4, n))
    y0 = x[0] + x[1] + rng.normal(size=n)
    trend = x[0] + x[2] + rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x[0] + x[2])))).astype(int)
    return {"y1": y0 + trend + d, "y0": y0, "d": d, "covariates": np.column_stack([np.ones(n), x.T])}


@pytest.fixture(scope="module")
def rc_planted_effect_arrays():
    """Repeated cross-section arrays with correct score and outcome models whose treated rows gain exactly one."""
    rng = np.random.default_rng(42)
    n = 20000
    x = rng.normal(size=(4, n))
    post = rng.binomial(1, 0.5, n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-(x[0] + x[2])))).astype(int)
    y = 1 + np.array([2.0, 3.0, 4.0, 5.0]) @ x + post + d * post + rng.normal(size=n)
    return {"y": y, "post": post, "d": d, "covariates": np.column_stack([np.ones(n), x.T])}


@pytest.fixture(scope="module")
def separated_two_period_arrays():
    """Panel and repeated cross-section outcomes whose covariate separates the treated units from the others."""
    rng = np.random.default_rng(42)
    n = 100
    x = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x[:, 1] > 0).astype(int)
    post = rng.binomial(1, 0.5, n)
    y = x @ [1, 0.5] + 2 * d * post + rng.standard_normal(n)
    y0 = x @ [1, 0.5] + rng.standard_normal(n)
    return {
        "panel": {"y1": y0 + 2 * d + rng.standard_normal(n), "y0": y0, "d": d, "covariates": x},
        "repeated_cross_sections": {"y": y, "post": post, "d": d, "covariates": x},
    }


@pytest.fixture(scope="module")
def panel_cell_arrays():
    """Panel arrays whose covariates indicate three cells and whose third cell has one comparison unit."""
    rng = np.random.default_rng(12)
    cell = np.repeat([0, 1, 2], [120, 120, 31])
    d = np.r_[rng.uniform(size=240) < np.repeat([0.3, 0.6], 120), np.ones(30), 0].astype(int)
    y0 = cell + rng.normal(size=cell.size)
    y1 = y0 + 1 + 0.5 * cell + d * (1 + cell) + 3 * (1 - d) * (cell == 2) + rng.normal(size=cell.size)
    covariates = np.column_stack([np.ones(cell.size), cell == 1, cell == 2]).astype(float)
    return {"y1": y1, "y0": y0, "d": d, "covariates": covariates, "i_weights": rng.uniform(0.5, 2, size=cell.size)}


@pytest.fixture(scope="module")
def rc_cell_arrays():
    """Repeated cross-sections whose covariates indicate three cells and whose third cell has two comparison rows."""
    rng = np.random.default_rng(13)
    cell = np.repeat([0, 1, 2], [160, 160, 62])
    post = np.r_[rng.uniform(size=320) < 0.5, np.tile([0, 1], 31)].astype(int)
    d = np.r_[rng.uniform(size=320) < np.repeat([0.3, 0.6], 160), np.ones(60), 0, 0].astype(int)
    trend = cell + post * (1 + 0.5 * cell) + d * (0.5 + post * (1 + cell)) + 3 * (1 - d) * (cell == 2) * post
    covariates = np.column_stack([np.ones(cell.size), cell == 1, cell == 2]).astype(float)
    return {
        "y": trend + rng.normal(size=cell.size),
        "post": post,
        "d": d,
        "covariates": covariates,
        "i_weights": rng.uniform(0.5, 2, size=cell.size),
    }


@pytest.fixture
def cell_panel_att():
    """Compute the closed-form IPW ATT of panel arrays whose covariates indicate cells."""

    def att(y1, y0, d, covariates, i_weights, trim_level=0.995, normalized=False):
        cell = np.rint(covariates[:, 1:] @ np.arange(1, covariates.shape[1])).astype(int)
        treated = np.bincount(cell, weights=i_weights * d)
        controls = np.bincount(cell, weights=i_weights * (1 - d))
        control_means = np.bincount(cell, weights=i_weights * (1 - d) * (y1 - y0)) / controls
        kept = treated / (treated + controls) < trim_level
        counterfactual = np.sum((treated * control_means)[kept]) / np.sum(treated[kept] if normalized else treated)
        return np.sum(i_weights * d * (y1 - y0)) / np.sum(treated) - counterfactual

    return att


@pytest.fixture
def cell_panel_inf_func():
    """Compute the influence function of the untrimmed IPW ATT of panel arrays whose covariates indicate cells."""

    def inf_func(y1, y0, d, covariates, i_weights):
        cell = np.rint(covariates[:, 1:] @ np.arange(1, covariates.shape[1])).astype(int)
        treated = np.bincount(cell, weights=i_weights * d)
        controls = np.bincount(cell, weights=i_weights * (1 - d))
        gaps = y1 - y0 - (np.bincount(cell, weights=i_weights * (1 - d) * (y1 - y0)) / controls)[cell]
        att = np.sum(i_weights * d * gaps) / np.sum(treated)
        return d.size * i_weights * (d * (gaps - att) - (1 - d) * (treated / controls)[cell] * gaps) / np.sum(treated)

    return inf_func


@pytest.fixture
def cell_rc_att():
    """Compute the closed-form IPW ATT of repeated cross-section arrays whose covariates indicate cells."""

    def att(y, post, d, covariates, i_weights, trim_level=0.995, normalized=False):
        cell = np.rint(covariates[:, 1:] @ np.arange(1, covariates.shape[1])).astype(int)
        ps = (np.bincount(cell, weights=i_weights * d) / np.bincount(cell, weights=i_weights))[cell]
        weights = i_weights * np.where(d == 1, 1.0, (ps < trim_level) * ps / (1 - ps))
        if normalized:
            means = [np.average(y, weights=weights * (d == g) * (post == t)) for g in (1, 0) for t in (1, 0)]
            return means[0] - means[1] - means[2] + means[3]
        lam = np.sum(i_weights * post) / np.sum(i_weights)
        return np.sum(weights * (2 * d - 1) * (post - lam) / (lam * (1 - lam)) * y) / np.sum(i_weights * d)

    return att


@pytest.fixture
def preset_bootstrap_weights(monkeypatch):
    """Hand the weighted bootstraps the rows of a preset matrix as their exponential draws."""

    def install(draws):
        rows = iter(draws)

        class PresetWeights:
            def __init__(self, seed=None):
                pass

            def exponential(self, scale=1.0, size=None):
                return next(rows)

        monkeypatch.setattr(np.random, "RandomState", PresetWeights)
        monkeypatch.setattr(np.random, "default_rng", PresetWeights)

    return install


@pytest.fixture(scope="module")
def wide_rc_two_period_data():
    """Two-period repeated cross-sections from gen_did_scalable with an intercept and twenty covariates."""
    simulated = gen_did_scalable(
        n=1000, dgp_type=1, n_periods=2, n_cohorts=1, n_covariates=20, panel=False, random_state=42
    )
    data = simulated["data"]
    return {
        "y": data["y"].to_numpy(),
        "post": (data["time"] == 2).cast(pl.Int64).to_numpy(),
        "d": (data["group"] == 2).cast(pl.Int64).to_numpy(),
        "x": np.column_stack([np.ones(data.height), data.select([f"cov{j}" for j in range(1, 21)]).to_numpy()]),
    }


@pytest.fixture
def aipw_panel_arrays():
    """Panel arguments of aipw_did_panel with three treated and three comparison units of unit weight."""
    return {
        "delta_y": np.array([1.0, 2.0, 0.0, 1.0, 3.0, 2.0]),
        "d": np.array([1, 1, 0, 0, 1, 0]),
        "ps": np.array([0.6, 0.7, 0.3, 0.4, 0.65, 0.35]),
        "out_reg": np.array([0.5, 0.6, 0.2, 0.3, 0.55, 0.25]),
        "i_weights": np.ones(6),
    }


@pytest.fixture
def aipw_rc_imp1_arrays():
    """Repeated cross-sections for aipw_did_rc_imp1 with two comparison rows and three treated rows per period."""
    return {
        "y": np.array([12.0, 9.0, 16.0, 22.0, 13.0, 17.0, 20.0, 21.0, 15.0, 25.0]),
        "post": np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1]),
        "d": np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
        "ps": np.array([0.5, 0.75, 0.25, 0.8, 0.6, 0.65, 0.58, 0.62, 0.55, 0.68]),
        "out_reg": np.array([10.0, 11.0, 15.0, 16.0, 10.0, 12.0, 13.0, 17.0, 14.0, 15.0]),
        "i_weights": np.ones(10),
    }


@pytest.fixture
def aipw_rc_imp2_arrays(aipw_rc_imp1_arrays):
    """The aipw_rc_imp1_arrays rows with outcome regressions of both groups in both periods for aipw_did_rc_imp2."""
    shared = {name: aipw_rc_imp1_arrays[name] for name in ("y", "post", "d", "ps", "i_weights")}
    return {
        **shared,
        "out_y_treat_post": np.array([14.0, 16.0, 18.0, 20.0, 16.0, 19.0, 14.0, 20.0, 22.0, 20.0]),
        "out_y_treat_pre": np.array([13.0, 12.0, 14.0, 18.0, 11.0, 14.0, 15.0, 15.0, 17.0, 17.0]),
        "out_y_cont_post": np.array([13.0, 14.0, 15.0, 16.0, 14.0, 15.0, 13.0, 17.0, 16.0, 15.0]),
        "out_y_cont_pre": np.array([10.0, 11.0, 12.0, 13.0, 10.0, 12.0, 11.0, 13.0, 14.0, 12.0]),
    }
