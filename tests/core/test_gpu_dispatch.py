"""Tests verifying GPU dispatch produces identical results to the CPU/statsmodels path."""

import importlib
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from moderndid.cupy.backend import set_backend
from moderndid.cupy.regression import cupy_logistic_irls, cupy_wls
from tests.helpers import importorskip

np = importorskip("numpy")
sm = importorskip("statsmodels.api")


class _NumpyAsGpu:
    def __getattr__(self, name):
        return getattr(np, name)


_FAKE_GPU = _NumpyAsGpu()
_REGRESSION_MOD = "moderndid.cupy.regression"


def _patch_backend(*modules):
    stack = ExitStack()
    for m in modules:
        stack.enter_context(patch(f"{m}.get_backend", return_value=_FAKE_GPU))
    return stack


def _generate_panel_data(rng, n=300):
    x1 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1])
    prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1)))
    d = rng.binomial(1, prob).astype(float)
    y0 = 1.0 + 0.5 * x1 + rng.standard_normal(n) * 0.5
    y1 = y0 + 2.0 * d + rng.standard_normal(n) * 0.5
    return y1, y0, d, covariates, np.ones(n)


def _generate_rc_data(rng, n=400):
    x1 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1])
    prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1)))
    d = rng.binomial(1, prob).astype(float)
    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    y = 1.0 + 0.5 * x1 + 1.0 * post + 2.0 * d * post + rng.standard_normal(n) * 0.5
    return y, post, d, covariates, np.ones(n)


@pytest.fixture(autouse=True)
def _numpy_backend():
    set_backend("numpy")
    yield
    set_backend("numpy")


@pytest.mark.parametrize(
    "seed,n,k,weighted",
    [
        (42, 200, 3, False),
        (123, 300, 4, True),
        (77, 250, 2, True),
    ],
)
def test_cupy_wls_vs_statsmodels(seed, n, k, weighted):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, k - 1))])
    beta_true = rng.standard_normal(k)
    y = X @ beta_true + rng.standard_normal(n) * 0.3
    w = rng.uniform(0.5, 3.0, n) if weighted else np.ones(n)

    beta_cupy, _ = cupy_wls(y, X, w)
    beta_sm = sm.WLS(y, X, weights=w).fit().params

    np.testing.assert_allclose(beta_cupy, beta_sm, rtol=1e-12)


def test_cupy_wls_vs_glm_gaussian():
    rng = np.random.default_rng(77)
    n = 250
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta_true = np.array([2.0, 1.5])
    y = X @ beta_true + rng.standard_normal(n) * 0.4
    w = rng.uniform(0.2, 2.0, n)

    beta_cupy, _ = cupy_wls(y, X, w)
    beta_glm = sm.GLM(y, X, family=sm.families.Gaussian(link=sm.families.links.Identity()), var_weights=w).fit().params

    np.testing.assert_allclose(beta_cupy, beta_glm, rtol=1e-12)


@pytest.mark.parametrize(
    "seed,n,k,weighted",
    [
        (42, 500, 2, False),
        (99, 400, 3, True),
        (55, 600, 6, True),
    ],
)
def test_cupy_logistic_irls_vs_statsmodels(seed, n, k, weighted):
    rng = np.random.default_rng(seed)
    X_raw = rng.standard_normal((n, k - 1))
    X = np.column_stack([np.ones(n), X_raw])
    beta_true = rng.standard_normal(k) * 0.5
    prob = 1 / (1 + np.exp(-(X @ beta_true)))
    y = rng.binomial(1, prob).astype(float)
    w = rng.uniform(0.3, 2.5, n) if weighted else np.ones(n)

    beta_cupy, mu_cupy = cupy_logistic_irls(y, X, w)
    glm_res = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()

    np.testing.assert_allclose(beta_cupy, glm_res.params, rtol=1e-8)
    np.testing.assert_allclose(mu_cupy, glm_res.predict(X), rtol=1e-8)


_PROPENSITY_ESTIMATORS = [
    pytest.param("moderndid.drdid.estimators.ipw_did_rc", "ipw_did_rc", "rc", id="ipw_did_rc"),
    pytest.param("moderndid.drdid.estimators.ipw_did_panel", "ipw_did_panel", "panel", id="ipw_did_panel"),
    pytest.param("moderndid.drdid.estimators.std_ipw_did_rc", "std_ipw_did_rc", "rc", id="std_ipw_did_rc"),
    pytest.param(
        "moderndid.drdid.estimators.std_ipw_did_panel",
        "std_ipw_did_panel",
        "panel",
        id="std_ipw_did_panel",
    ),
    pytest.param("moderndid.drdid.estimators.drdid_trad_rc", "drdid_trad_rc", "rc", id="drdid_trad_rc"),
]

_OUTCOME_ESTIMATORS = [
    pytest.param("moderndid.drdid.estimators.reg_did_rc", "reg_did_rc", "rc", id="reg_did_rc"),
    pytest.param("moderndid.drdid.estimators.reg_did_panel", "reg_did_panel", "panel", id="reg_did_panel"),
    pytest.param("moderndid.drdid.estimators.twfe_did_rc", "twfe_did_rc", "rc", id="twfe_did_rc"),
    pytest.param(
        "moderndid.drdid.estimators.twfe_did_panel",
        "twfe_did_panel",
        "panel",
        id="twfe_did_panel",
    ),
]


def _run_estimator_dispatch_test(module_path, fn_name, data_type, rtol_att, rtol_inf):
    mod = importlib.import_module(module_path)
    estimator = getattr(mod, fn_name)
    rng = np.random.default_rng(42)

    if data_type == "panel":
        data = _generate_panel_data(rng)
    else:
        data = _generate_rc_data(rng)

    cpu = estimator(*data, influence_func=True)

    with _patch_backend(module_path, _REGRESSION_MOD):
        gpu = estimator(*data, influence_func=True)

    np.testing.assert_allclose(gpu.att, cpu.att, rtol=rtol_att)
    np.testing.assert_allclose(gpu.se, cpu.se, rtol=rtol_att)
    np.testing.assert_allclose(gpu.att_inf_func, cpu.att_inf_func, rtol=rtol_inf)


@pytest.mark.parametrize("module_path,fn_name,data_type", _PROPENSITY_ESTIMATORS)
def test_propensity_dispatch(module_path, fn_name, data_type):
    _run_estimator_dispatch_test(module_path, fn_name, data_type, rtol_att=1e-8, rtol_inf=1e-7)


@pytest.mark.parametrize("module_path,fn_name,data_type", _OUTCOME_ESTIMATORS)
def test_outcome_regression_dispatch(module_path, fn_name, data_type):
    _run_estimator_dispatch_test(module_path, fn_name, data_type, rtol_att=1e-10, rtol_inf=1e-8)


def test_ddd_compute_pscore_rc_dispatch():
    from moderndid.didtriple.nuisance_rc import _compute_pscore_rc

    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1])
    weights = np.ones(n)
    prob = 1 / (1 + np.exp(-(0.3 + 0.5 * x1)))
    subgroup = np.where(rng.random(n) < prob, 4, 3).astype(float)
    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

    cpu = _compute_pscore_rc(subgroup, post, covariates, weights, comparison_subgroup=3)

    with _patch_backend("moderndid.didtriple.nuisance_rc", _REGRESSION_MOD):
        gpu = _compute_pscore_rc(subgroup, post, covariates, weights, comparison_subgroup=3)

    np.testing.assert_allclose(gpu.propensity_scores, cpu.propensity_scores, rtol=1e-8)
    np.testing.assert_array_equal(gpu.keep_ps, cpu.keep_ps)


def test_ddd_fit_ols_cell_dispatch():
    from moderndid.didtriple.nuisance_rc import _fit_ols_cell

    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    covariates = np.column_stack([np.ones(n), x1])
    weights = rng.uniform(0.5, 2.0, n)
    d = rng.binomial(1, 0.5, n).astype(int)
    post = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    y = 1.0 + 0.5 * x1 + rng.standard_normal(n) * 0.3

    cpu = _fit_ols_cell(
        y=y,
        post=post,
        d=d,
        covariates=covariates,
        weights=weights,
        pre=True,
        treat=False,
        n_features=2,
        comparison_subgroup=3,
    )

    with _patch_backend("moderndid.didtriple.nuisance_rc", _REGRESSION_MOD):
        gpu = _fit_ols_cell(
            y=y,
            post=post,
            d=d,
            covariates=covariates,
            weights=weights,
            pre=True,
            treat=False,
            n_features=2,
            comparison_subgroup=3,
        )

    np.testing.assert_allclose(gpu, cpu, rtol=1e-12)
