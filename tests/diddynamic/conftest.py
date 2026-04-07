"""Shared fixtures for diddynamic tests."""

import warnings

import numpy as np
import polars as pl
import pytest

from moderndid.core.preprocess import DynBalancingConfig, PreprocessDataBuilder
from moderndid.diddynamic.container import DynBalancingResult
from moderndid.diddynamic.dyn_balancing import dyn_balancing


def build_dyn_balancing(data, **config_kwargs):
    """Run the full builder pipeline for DynBalancing."""
    config = DynBalancingConfig(**config_kwargs)
    return PreprocessDataBuilder().with_data(data).with_config(config).validate().transform().build()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def base_config():
    """Base config kwargs for preprocessing tests."""
    return dict(yname="y", tname="time", idname="id", treatment_name="D", ds1=[0, 0, 1, 1], ds2=[0, 0, 0, 0])


@pytest.fixture
def simple_panel():
    """Balanced panel with 10 units and 4 periods."""
    rng = np.random.default_rng(42)
    n_units = 10
    n_periods = 4
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(n_units * n_periods)
    for i in range(n_units):
        if i < 5:
            treatment[i * n_periods + 2 : (i + 1) * n_periods] = 1.0
    y = rng.standard_normal(n_units * n_periods)
    x1 = rng.standard_normal(n_units * n_periods)
    x2 = rng.standard_normal(n_units * n_periods)
    cluster = np.repeat(np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3]), n_periods)

    return pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "D": treatment,
            "X1": x1,
            "X2": x2,
            "cluster_var": cluster,
        }
    )


@pytest.fixture
def unbalanced_panel():
    """Panel where unit 1 is missing period 3."""
    return pl.DataFrame(
        {
            "id": [0, 0, 0, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 1, 2, 3],
            "y": [1.0] * 8,
            "D": [0.0] * 8,
        }
    )


@pytest.fixture
def estimation_panel(rng):
    """Synthetic panel for estimation module tests (60 units, 3 periods, 3 covariates)."""
    n_units = 60
    n_periods = 3
    n_covariates = 3
    treatment = np.zeros((n_units, n_periods))
    for i in range(n_units // 2):
        treatment[i, 1:] = 1.0
    covariates = {t: rng.standard_normal((n_units, n_covariates)) for t in range(n_periods)}
    outcome = rng.standard_normal(n_units) + 0.5 * covariates[n_periods - 1][:, 0] + 2.0 * treatment[:, -1]
    ds = np.array([0.0, 1.0, 1.0])
    return outcome, treatment, covariates, ds


@pytest.fixture
def simple_qp_data(rng):
    """Small balanced data for direct QP testing."""
    n = 20
    p = 2
    x_all = rng.standard_normal((n, p))
    d_col = np.array([1.0] * 10 + [0.0] * 10)
    coef = np.zeros(p + 1)
    return x_all, d_col, coef


@pytest.fixture
def estimator_panel():
    """Panel with 60 units and 3 periods suitable for end-to-end estimation."""
    rng = np.random.default_rng(99)
    n_units = 60
    n_periods = 3
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(n_units * n_periods)
    for i in range(n_units // 2):
        treatment[i * n_periods + 1 : (i + 1) * n_periods] = 1.0
    y = rng.standard_normal(n_units * n_periods) + 2.0 * treatment
    x1 = rng.standard_normal(n_units * n_periods)
    x2 = rng.standard_normal(n_units * n_periods)
    cluster = np.repeat(np.arange(n_units) % 5, n_periods)

    return pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "D": treatment,
            "X1": x1,
            "X2": x2,
            "cluster_var": cluster,
        }
    )


@pytest.fixture
def sample_result():
    """DynBalancingResult for container and format tests."""
    return DynBalancingResult(
        att=2.345,
        var_att=0.015129,
        mu1=5.678,
        mu2=3.333,
        var_mu1=0.007921,
        var_mu2=0.007569,
        robust_quantile=3.84,
        gaussian_quantile=1.96,
        gammas={"ds1": np.ones(10) / 10, "ds2": np.ones(10) / 10},
        coefficients={"ds1": np.array([0.1, 0.2]), "ds2": np.array([0.3, 0.4])},
        imbalances={"ds1": 0.01, "ds2": 0.02},
        estimation_params={
            "n_obs": 500,
            "n_units": 250,
            "yname": "outcome",
            "balancing": "dcb",
            "method": "lasso_plain",
            "ds1": [1, 1],
            "ds2": [0, 0],
            "alpha": 0.05,
            "robust_quantile": True,
        },
    )


@pytest.fixture
def single_period_data(rng):
    """Gammas, predictions, not_nas, and outcome for a single-period setup."""
    n = 20
    gammas = np.zeros((n, 1))
    gammas[:10, 0] = 1.0 / 10
    predictions = rng.standard_normal((n, 1))
    not_nas = [np.arange(n)]
    y_t = predictions[:, 0] + rng.standard_normal(n) * 0.1
    return gammas, predictions, not_nas, y_t


@pytest.fixture
def multi_period_data(rng):
    """Gammas, predictions, not_nas, and outcome for a three-period setup."""
    n = 30
    n_periods = 3
    gammas = np.zeros((n, n_periods))
    gammas[:15, 0] = 1.0 / 15
    gammas[:15, 1] = 1.0 / 15
    gammas[:15, 2] = 1.0 / 15
    predictions = rng.standard_normal((n, n_periods))
    not_nas = [np.arange(n)] * n_periods
    y_t = predictions[:, -1] + rng.standard_normal(n) * 0.5
    return gammas, predictions, not_nas, y_t


@pytest.fixture
def history_result(estimator_panel):
    """History result with lag lengths 1, 2, 3 for estimation tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ds1 contains one element")
        return dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            histories_length=[1, 2, 3],
            xformla="~ X1",
            ub=20.0,
            grid_length=50,
            nfolds=3,
            adaptive_balancing=False,
        )


@pytest.fixture
def impulse_panel():
    """Panel with diverse treatment patterns including [1,0] over last 2 periods."""
    rng = np.random.default_rng(77)
    n_units = 80
    n_periods = 3
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(n_units * n_periods)
    # Units 0-19: pattern [0, 0, 0] (never treated)
    # Units 20-39: pattern [0, 1, 0] (impulse at period 2)
    for i in range(20, 40):
        treatment[i * n_periods + 1] = 1.0
    # Units 40-59: pattern [1, 0, 0] (impulse at period 1)
    for i in range(40, 60):
        treatment[i * n_periods] = 1.0
    # Units 60-79: pattern [0, 1, 1] (treated from period 2)
    for i in range(60, 80):
        treatment[i * n_periods + 1 : (i + 1) * n_periods] = 1.0
    y = rng.standard_normal(n_units * n_periods) + 1.5 * treatment
    x1 = rng.standard_normal(n_units * n_periods)
    return pl.DataFrame({"id": ids, "time": times, "y": y, "D": treatment, "X1": x1})


@pytest.fixture
def impulse_result(impulse_panel):
    """Impulse response result with history lengths 2 and 3."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ds1 contains one element")
        return dyn_balancing(
            data=impulse_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1, 1, 1],
            ds2=[0, 0, 0],
            histories_length=[2, 3],
            impulse_response=True,
            xformla="~ X1",
            ub=20.0,
            grid_length=50,
            nfolds=3,
            adaptive_balancing=False,
        )


@pytest.fixture
def het_result(estimator_panel):
    """Het result with final_periods=[2, 3] for estimation tests."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ds1 contains one element")
        return dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1],
            ds2=[0],
            final_periods=[2, 3],
            xformla="~ X1",
            ub=20.0,
            grid_length=50,
            nfolds=3,
            adaptive_balancing=False,
        )
