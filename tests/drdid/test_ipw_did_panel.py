"""Tests for IPW DiD estimator with panel data."""

import numpy as np
import pytest

from moderndid.drdid import ipw_did_panel


def test_ipw_did_panel_basic():
    np.random.seed(42)
    n = 200
    p = 3

    x = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
    d = np.random.binomial(1, 0.3, n)
    y0 = x @ [1, 0.5, -0.3] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert isinstance(result.se, float)
    assert result.se > 0
    assert result.lci < result.att < result.uci
    assert result.boots is None
    assert result.att_inf_func is None


def test_ipw_did_panel_with_weights():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5
    weights = np.random.exponential(1, n)

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=weights)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_panel_with_influence_func():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, influence_func=True)

    assert result.att_inf_func is not None
    assert len(result.att_inf_func) == n
    assert np.abs(np.mean(result.att_inf_func)) < 0.1


def test_ipw_did_panel_no_covariates():
    np.random.seed(42)
    n = 200
    d = np.random.binomial(1, 0.3, n)
    y0 = 1 + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=None)

    assert isinstance(result.att, float)
    assert result.se > 0


def test_ipw_did_panel_bootstrap_weighted():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, boot=True, boot_type="weighted", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10
    assert not np.all(np.isnan(result.boots))


def test_ipw_did_panel_bootstrap_multiplier():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, boot=True, boot_type="multiplier", nboot=10)

    assert result.boots is not None
    assert len(result.boots) == 10


def test_ipw_did_panel_invalid_inputs():
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y0 = np.random.randn(n)
    y1 = np.random.randn(n)
    d = np.random.binomial(1, 0.5, n)

    with pytest.raises(ValueError, match="i_weights must be non-negative"):
        ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, i_weights=-np.ones(n))


def test_ipw_did_panel_edge_cases():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    y0 = np.random.randn(n)
    y1 = np.random.randn(n)

    d_all_treated = np.ones(n, dtype=int)
    with pytest.raises(ValueError, match="No control units found"):
        ipw_did_panel(y1=y1, y0=y0, d=d_all_treated, covariates=x)

    d_all_control = np.zeros(n, dtype=int)
    with pytest.raises(ValueError, match="No treated units found"):
        ipw_did_panel(y1=y1, y0=y0, d=d_all_control, covariates=x)


def test_ipw_did_panel_extreme_pscore():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.linspace(-10, 10, n)])
    d = (x[:, 1] > 0).astype(int)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    with pytest.warns(UserWarning):
        result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert isinstance(result.att, float)


def test_ipw_did_panel_no_effective_treated():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.zeros(n, dtype=int)
    d[x[:, 1] > 2.5] = 1

    if np.sum(d) == 0:
        d[-3:] = 1
        x[-3:, 1] = 3.0

    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, trim_level=0.99)

    assert isinstance(result.att, float) or np.isnan(result.att)


def test_ipw_did_panel_trim_level():
    np.random.seed(42)
    n = 200
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result1 = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, trim_level=0.99)
    result2 = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, trim_level=0.95)

    assert isinstance(result1.att, float)
    assert isinstance(result2.att, float)


def test_ipw_did_panel_args_output():
    np.random.seed(42)
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x, boot=True, nboot=10, trim_level=0.99)

    assert result.args["panel"] is True
    assert result.args["normalized"] is False
    assert result.args["boot"] is True
    assert result.args["boot_type"] == "weighted"
    assert result.args["nboot"] == 10
    assert result.args["type"] == "ipw"
    assert result.args["trim_level"] == 0.99


def test_ipw_did_panel_reproducibility():
    n = 100
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    np.random.seed(42)
    result1 = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    np.random.seed(42)
    result2 = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert result1.att == result2.att
    assert result1.se == result2.se


def test_ipw_did_panel_multicollinear_covariates():
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = 2 * x1
    x = np.column_stack([np.ones(n), x1, x2])
    d = np.random.binomial(1, 0.5, n)
    y0 = 1 + 0.5 * x1 + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)
    assert isinstance(result.att, float)
    assert not np.isnan(result.att)
    assert result.se > 0


def test_ipw_did_panel_arrays_vs_lists():
    np.random.seed(42)
    n = 50
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.random.binomial(1, 0.5, n)
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result_array = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    result_list = ipw_did_panel(y1=y1.tolist(), y0=y0.tolist(), d=d.tolist(), covariates=x.tolist())

    assert np.isclose(result_array.att, result_list.att, rtol=1e-10)
    assert np.isclose(result_array.se, result_list.se, rtol=1e-10)


def test_ipw_did_panel_small_sample_warning():
    np.random.seed(42)
    n = 10
    x = np.column_stack([np.ones(n), np.random.randn(n)])
    d = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y0 = x @ [1, 0.5] + np.random.randn(n)
    y1 = y0 + 2 * d + np.random.randn(n) * 0.5

    result = ipw_did_panel(y1=y1, y0=y0, d=d, covariates=x)

    assert isinstance(result.att, float)
    assert result.se > 0
