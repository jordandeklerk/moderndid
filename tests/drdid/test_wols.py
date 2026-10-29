"""Tests for weighted OLS regression functions."""

import re
import warnings

import numpy as np
import pytest

from moderndid import ols_panel, wols_panel, wols_rc
from moderndid.cupy.backend import use_backend
from moderndid.drdid.estimators.wols import _has_collinear_columns, ols_rc


def test_ols_panel_fits_least_squares_on_the_comparison_units():
    delta_y = np.array([1.5, 2.0, 0.5, 1.0, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(42).standard_normal((10, 3))
    i_weights = np.array([0.5, 1.5, 0.0, 1.2, 0.9, 1.1, 1.3, 0.7, 1.6, 0.6])
    root = np.sqrt(i_weights[d == 0])
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ols_panel(delta_y, d, x, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)
    assert not caught


def test_wols_panel_weights_the_comparison_units_by_their_propensity_odds():
    delta_y = np.array([1.5, 2.0, 0.5, 1.0, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(42).standard_normal((10, 3))
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.35, 0.65, 0.25, 0.45, 0.55, 0.38])
    i_weights = np.ones(10)
    root = np.sqrt(ps[d == 0] / (1 - ps[d == 0]))
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wols_panel(delta_y, d, x, ps, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)
    assert not caught


def test_wols_panel_multiplies_the_odds_by_the_observation_weights():
    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
    d = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(123).standard_normal((8, 2))
    ps = np.array([0.6, 0.3, 0.4, 0.7, 0.35, 0.25, 0.55, 0.45])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7])
    root = np.sqrt(i_weights[d == 0] * ps[d == 0] / (1 - ps[d == 0]))
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wols_panel(delta_y, d, x, ps, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)
    assert not caught


def test_wols_panel_no_control_units():
    n_units = 5
    n_features = 2

    delta_y = np.array([1.0, 2.0, 1.5, 1.2, 0.8])
    d = np.ones(n_units, dtype=int)
    x = np.random.default_rng(456).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.7, 0.65, 0.8, 0.75])
    i_weights = np.ones(n_units)

    message = "No units found in the comparison group. Pass data in which some units have a treatment indicator of 0."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        wols_panel(delta_y, d, x, ps, i_weights)


def test_wols_panel_few_control_units():
    delta_y = np.array([1.0, 2.0, 1.5, 1.2, 0.8, 0.5, 0.9, 1.1, 0.7, 0.6])
    d = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
    x = np.random.default_rng(789).standard_normal((10, 2))
    ps = np.array([0.6, 0.7, 0.65, 0.8, 0.75, 0.72, 0.3, 0.25, 0.35, 0.4])
    i_weights = np.ones(10)
    root = np.sqrt(ps[d == 0] / (1 - ps[d == 0]))
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]
    message = "Only 4 control units available. Results may be unreliable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = wols_panel(delta_y, d, x, ps, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)


@pytest.mark.parametrize(
    ("n_controls", "messages"), [(4, ["Only 4 control units available. Results may be unreliable."]), (5, [])]
)
def test_ols_panel_warns_below_five_comparison_units(n_controls, messages):
    rng = np.random.default_rng(16)
    d = np.r_[np.ones(6), np.zeros(n_controls)].astype(int)
    x = np.column_stack([np.ones(6 + n_controls), rng.standard_normal(6 + n_controls)])
    delta_y = rng.standard_normal(6 + n_controls)
    i_weights = rng.uniform(0.5, 2.0, 6 + n_controls)
    root = np.sqrt(i_weights[d == 0])
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ols_panel(delta_y, d, x, i_weights)

    assert [str(w.message) for w in caught] == messages
    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)


def test_wols_panel_ps_one_for_control():
    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
    d = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=int)
    x = np.random.default_rng(111).standard_normal((8, 2))
    ps = np.array([0.6, 1.0, 0.4, 0.7, 0.35, 0.3, 0.45, 0.5])
    i_weights = np.ones(8)

    with pytest.raises(ValueError, match="Propensity score is 1 for some control units"):
        wols_panel(delta_y, d, x, ps, i_weights)


def test_wols_panel_accepts_a_score_of_one_for_a_treated_unit():
    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
    d = np.array([1, 0, 0, 1, 0, 0, 0, 1], dtype=int)
    x = np.random.default_rng(111).standard_normal((8, 2))
    ps = np.array([1.0, 0.2, 0.4, 0.7, 0.35, 0.3, 0.45, 0.5])
    i_weights = np.ones(8)
    root = np.sqrt(ps[d == 0] / (1 - ps[d == 0]))
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]

    result = wols_panel(delta_y, d, x, ps, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)


def test_wols_panel_extreme_weight_ratios():
    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.4, 1.1])
    d = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=int)
    x = np.random.default_rng(222).standard_normal((8, 2))
    ps = np.array([0.6, 0.999999, 0.001, 0.4, 0.7, 0.35, 0.5, 0.45])
    i_weights = np.ones(8)
    root = np.sqrt(ps[d == 0] / (1 - ps[d == 0]))
    expected = np.linalg.lstsq(x[d == 0] * root[:, None], delta_y[d == 0] * root, rcond=None)[0]
    message = "Extreme weight ratios detected. Results may be numerically unstable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = wols_panel(delta_y, d, x, ps, i_weights)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)


@pytest.mark.parametrize("regression", [ols_panel, ols_rc])
def test_ols_regressions_warn_about_extreme_observation_weights(regression):
    rng = np.random.default_rng(17)
    post = np.repeat([0, 1, 0, 1], 6)
    d = np.repeat([0, 0, 1, 1], 6)
    x = np.column_stack([np.ones(24), rng.standard_normal(24)])
    y = rng.standard_normal(24)
    i_weights = np.r_[1e7, np.ones(23)]
    rows = (d == 0) if regression is ols_panel else (d == 0) & (post == 0)
    root = np.sqrt(i_weights[rows])
    expected = np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]
    spec = {"delta_y": y} if regression is ols_panel else {"y": y, "post": post, "pre": True, "treat": False}
    message = "Extreme weight ratios detected. Results may be numerically unstable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = regression(d=d, x=x, i_weights=i_weights, **spec)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)


def test_wols_panel_multicollinearity():
    n_units = 10
    n_features = 3

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=int)

    x = np.zeros((n_units, n_features))
    x[:, 0] = np.random.default_rng(333).standard_normal(n_units)
    x[:, 1] = x[:, 0] * 2
    x[:, 2] = x[:, 0] + x[:, 1]
    ps = np.array([0.6, 0.3, 0.4, 0.35, 0.7, 0.25, 0.45, 0.38, 0.65, 0.32])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="^The regression design matrix of the comparison group is singular\\."):
        wols_panel(delta_y, d, x, ps, i_weights)


@pytest.mark.parametrize(
    "invalid_input,index",
    [
        ([1.0], 0),
        ([1.0], 1),
        ([[1.0]], 2),
        ([1.0], 3),
        ([1.0], 4),
    ],
)
def test_wols_panel_type_errors(invalid_input, index):
    valid_1d = np.array([1.0])
    valid_2d = np.array([[1.0]])
    inputs = [valid_1d, valid_1d, valid_2d, valid_1d, valid_1d]
    inputs[index] = invalid_input

    with pytest.raises(TypeError, match="All inputs must be NumPy arrays"):
        wols_panel(*inputs)


def test_wols_panel_dimension_errors():
    valid_1d = np.array([1.0, 2.0])
    valid_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    invalid_2d = np.array([[1.0], [2.0]])
    invalid_3d = np.array([[[1.0]]])

    with pytest.raises(ValueError, match="delta_y, d, ps, and i_weights must be 1-dimensional"):
        wols_panel(invalid_2d, valid_1d, valid_2d, valid_1d, valid_1d)

    with pytest.raises(ValueError, match="x must be a 2-dimensional array"):
        wols_panel(valid_1d, valid_1d, valid_1d, valid_1d, valid_1d)

    with pytest.raises(ValueError, match="x must be a 2-dimensional array"):
        wols_panel(valid_1d, valid_1d, invalid_3d, valid_1d, valid_1d)


def test_wols_panel_shape_mismatch():
    delta_y = np.array([1.0, 2.0])
    d = np.array([1])
    x = np.array([[1.0], [2.0]])
    ps = np.array([0.5, 0.5])
    i_weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wols_panel(delta_y, d, x, ps, i_weights)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wols_panel(delta_y, np.array([1, 0]), np.ones((3, 1)), ps, i_weights)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize(
    ("pre", "treat", "sizes", "message"),
    [
        (
            True,
            False,
            [2, 4, 4, 4],
            "The comparison group in the pre-period has fewer observations (2) than regression coefficients (3). "
            "Pass fewer covariates.",
        ),
        (
            False,
            False,
            [4, 2, 4, 4],
            "The comparison group in the post-period has fewer observations (2) than regression coefficients (3). "
            "Pass fewer covariates.",
        ),
        (
            True,
            True,
            [4, 4, 2, 4],
            "The treated group in the pre-period has fewer observations (2) than regression coefficients (3). "
            "Pass fewer covariates. The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the "
            "treated group and accept this design.",
        ),
        (
            False,
            True,
            [4, 4, 4, 2],
            "The treated group in the post-period has fewer observations (2) than regression coefficients (3). "
            "Pass fewer covariates. The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the "
            "treated group and accept this design.",
        ),
    ],
)
def test_rc_regressions_reject_a_cell_with_fewer_observations_than_coefficients(regression, pre, treat, sizes, message):
    post = np.repeat([0, 1, 0, 1], sizes)
    d = np.repeat([0, 0, 1, 1], sizes)
    rng = np.random.default_rng(5)
    x = np.column_stack([np.ones(post.size), rng.standard_normal((post.size, 2))])
    extra = {"ps": np.full(post.size, 0.5)} if regression is wols_rc else {}

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(
            y=rng.standard_normal(post.size),
            post=post,
            d=d,
            x=x,
            i_weights=np.ones(post.size),
            pre=pre,
            treat=treat,
            **extra,
        )


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            [0.0, 0.0, 1.0, 1.0],
            "The comparison group in the pre-period has fewer observations with positive weight (2) than "
            "regression coefficients (3). Pass fewer covariates.",
        ),
        (
            [0.0, 0.0, 0.0, 0.0],
            "The comparison group in the pre-period has no observations with positive weight. Pass positive "
            "weights for some of its observations.",
        ),
    ],
)
def test_rc_regressions_count_only_observations_with_positive_weight(regression, weights, message):
    post = np.repeat([0, 1, 0, 1], 4)
    d = np.repeat([0, 0, 1, 1], 4)
    rng = np.random.default_rng(6)
    x = np.column_stack([np.ones(16), rng.standard_normal((16, 2))])
    extra = {"ps": np.full(16, 0.5)} if regression is wols_rc else {}

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(
            y=rng.standard_normal(16),
            post=post,
            d=d,
            x=x,
            i_weights=np.r_[weights, np.ones(12)],
            pre=True,
            treat=False,
            **extra,
        )


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize(
    ("n_rows", "messages"), [(2, ["Only 2 units available for regression. Results may be unreliable."]), (3, [])]
)
def test_rc_regressions_warn_below_three_rows_and_fit_a_cell_with_as_many_rows_as_coefficients(
    regression, n_rows, messages
):
    post = np.repeat([0, 1, 0, 1], [n_rows, 4, 4, 4])
    d = np.repeat([0, 0, 1, 1], [n_rows, 4, 4, 4])
    rng = np.random.default_rng(14)
    x = np.column_stack([np.ones(post.size), rng.standard_normal(post.size)])
    y = rng.standard_normal(post.size)
    ps = rng.uniform(0.2, 0.8, post.size)
    root = np.sqrt(ps[:n_rows] / (1 - ps[:n_rows]) if regression is wols_rc else np.ones(n_rows))
    expected = np.linalg.lstsq(x[:n_rows] * root[:, None], y[:n_rows] * root, rcond=None)[0]
    extra = {"ps": ps} if regression is wols_rc else {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = regression(y=y, post=post, d=d, x=x, i_weights=np.ones(post.size), pre=True, treat=False, **extra)

    assert [str(w.message) for w in caught] == messages
    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)


def test_wols_rc_fits_the_treated_cell_with_the_observation_weights_alone():
    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    x = np.random.default_rng(555).standard_normal((8, 2))
    ps = np.array([0.6, 0.4, 0.65, 0.35, 0.55, 0.45, 0.58, 0.42])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7])
    rows = (d == 1) & (post == 0)
    root = np.sqrt(i_weights[rows])
    expected = np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=True)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)
    assert not caught


def test_wols_rc_pre_parameter_required():
    n_units = 4
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0])
    post = np.array([0, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 1, 1], dtype=int)
    x = np.random.default_rng(666).standard_normal((n_units, n_features))
    ps = np.array([0.4, 0.5, 0.6, 0.7])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="pre parameter must be specified"):
        wols_rc(y, post, d, x, ps, i_weights, pre=None, treat=False)


def test_wols_rc_no_units_in_subset():
    n_units = 6
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0])
    post = np.ones(n_units, dtype=int)
    d = np.zeros(n_units, dtype=int)
    x = np.random.default_rng(777).standard_normal((n_units, n_features))
    ps = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.42])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="^No units found in the comparison group in the pre-period\\."):
        wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)


def test_wols_rc_ps_one_in_subset():
    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0, 11.0])
    post = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    x = np.random.default_rng(999).standard_normal((9, 2))
    ps = np.array([1.0, 0.5, 0.4, 0.45, 0.3, 0.6, 0.7, 0.65, 0.55])
    i_weights = np.ones(9)

    with pytest.raises(ValueError, match="Propensity score is 1 for some units in subset"):
        wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)


def test_wols_rc_ignores_a_score_of_one_outside_the_fitted_cell():
    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0, 11.0])
    post = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    x = np.random.default_rng(999).standard_normal((9, 2))
    ps = np.array([0.2, 0.5, 0.4, 1.0, 0.3, 1.0, 0.7, 0.65, 0.55])
    i_weights = np.ones(9)
    root = np.sqrt(ps[:3] / (1 - ps[:3]))
    expected = np.linalg.lstsq(x[:3] * root[:, None], y[:3] * root, rcond=None)[0]

    result = wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)


def test_wols_rc_extreme_weights():
    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0, 13.0, 17.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int)
    x = np.random.default_rng(1111).standard_normal((10, 2))
    ps = np.array([0.001, 0.5, 0.4, 0.999999, 0.6, 0.7, 0.65, 0.72, 0.3, 0.45])
    i_weights = np.ones(10)
    rows = (d == 0) & (post == 1)
    root = np.sqrt(ps[rows] / (1 - ps[rows]))
    expected = np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]
    message = "Extreme weight ratios detected. Results may be numerically unstable."

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = wols_rc(y, post, d, x, ps, i_weights, pre=False, treat=False)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
def test_rc_regressions_reject_collinear_covariates(regression):
    post = np.repeat([0, 1, 0, 1], 4)
    d = np.repeat([0, 0, 1, 1], 4)
    x1 = np.random.default_rng(2222).standard_normal(16)
    x = np.column_stack([np.ones(16), x1, 2 * x1])
    extra = {"ps": np.full(16, 0.5)} if regression is wols_rc else {}
    message = (
        "The regression design matrix of the comparison group in the pre-period is singular. Consider removing "
        "covariates that are constant or collinear within that group and period."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(y=x1 + 1, post=post, d=d, x=x, i_weights=np.ones(16), pre=True, treat=False, **extra)


@pytest.mark.parametrize("regression", [ols_panel, wols_panel])
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            [1.0, 1.0],
            "The comparison group has fewer units (2) than regression coefficients (3). Pass fewer covariates.",
        ),
        (
            [0.0, 0.0, 1.0, 1.0],
            "The comparison group has fewer units with positive weight (2) than regression coefficients (3). "
            "Pass fewer covariates.",
        ),
    ],
)
def test_panel_regressions_reject_a_comparison_group_with_fewer_units_than_coefficients(regression, weights, message):
    n = 6 + len(weights)
    rng = np.random.default_rng(7)
    x = np.column_stack([np.ones(n), rng.standard_normal((n, 2))])
    extra = {"ps": np.full(n, 0.5)} if regression is wols_panel else {}

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(
            delta_y=rng.standard_normal(n),
            d=np.r_[np.ones(6), np.zeros(len(weights))],
            x=x,
            i_weights=np.r_[np.ones(6), weights],
            **extra,
        )


@pytest.mark.parametrize(
    "invalid_input,index",
    [
        ([1.0], 0),
        ([1.0], 1),
        ([1.0], 2),
        ([[1.0]], 3),
        ([1.0], 4),
        ([1.0], 5),
    ],
)
def test_wols_rc_type_errors(invalid_input, index):
    valid_1d = np.array([1.0])
    valid_2d = np.array([[1.0]])
    inputs = [valid_1d, valid_1d, valid_1d, valid_2d, valid_1d, valid_1d]
    inputs[index] = invalid_input

    with pytest.raises(TypeError, match="All inputs must be NumPy arrays"):
        wols_rc(*inputs, pre=True)


def test_wols_rc_dimension_errors():
    valid_1d = np.array([1.0, 2.0])
    valid_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
    invalid_2d = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="y, post, d, ps, and i_weights must be 1-dimensional"):
        wols_rc(invalid_2d, valid_1d, valid_1d, valid_2d, valid_1d, valid_1d, pre=True)

    with pytest.raises(ValueError, match="x must be a 2-dimensional array"):
        wols_rc(valid_1d, valid_1d, valid_1d, valid_1d, valid_1d, valid_1d, pre=True)


def test_wols_rc_shape_mismatch():
    y = np.array([1.0, 2.0])
    post = np.array([1])
    d = np.array([1, 0])
    x = np.array([[1.0], [2.0]])
    ps = np.array([0.5, 0.5])
    i_weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wols_rc(y, post, d, x, ps, i_weights, pre=True)

    with pytest.raises(ValueError, match="All arrays must have the same number of observations"):
        wols_rc(y, np.array([0, 1]), d, np.ones((3, 1)), ps, i_weights, pre=True)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize("pre,treat", [(True, True), (True, False), (False, True), (False, False)])
def test_rc_regressions_fit_each_cell_on_its_own_rows_and_weights(regression, pre, treat):
    rng = np.random.default_rng(3333)
    post = np.tile([0, 1], 20)
    d = np.repeat([0, 1], 20)
    x = np.column_stack([np.ones(40), rng.standard_normal((40, 2))])
    y = 15 + x @ np.array([0.0, 1.0, -1.0]) + rng.standard_normal(40)
    ps = rng.uniform(0.2, 0.8, 40)
    i_weights = rng.uniform(0.5, 2.0, 40)
    rows = (d == int(treat)) & (post == int(not pre))
    weights = i_weights * (ps / (1 - ps) if regression is wols_rc and not treat else 1.0)
    root = np.sqrt(weights[rows])
    expected = np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]
    extra = {"ps": ps} if regression is wols_rc else {}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = regression(y=y, post=post, d=d, x=x, i_weights=i_weights, pre=pre, treat=treat, **extra)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)
    assert not caught


@pytest.mark.parametrize("regression", [ols_panel, wols_panel, ols_rc, wols_rc])
def test_regressions_match_least_squares_with_twenty_covariates_and_few_comparison_rows(regression):
    rng = np.random.default_rng(21)
    d = np.r_[np.zeros(24), np.ones(60)].astype(int)
    post = np.r_[np.zeros(24), np.tile([0, 1], 30)].astype(int)
    x = np.column_stack([np.ones(84), 1 + rng.standard_normal((84, 20))])
    y = x @ rng.standard_normal(21) + rng.standard_normal(84)
    ps = rng.uniform(0.2, 0.8, 84)
    i_weights = rng.uniform(0.5, 2.0, 84)
    weighted = regression in (wols_panel, wols_rc)
    root = np.sqrt(i_weights[:24] * (ps[:24] / (1 - ps[:24]) if weighted else 1.0))
    expected = np.linalg.lstsq(x[:24] * root[:, None], y[:24] * root, rcond=None)[0]
    spec = {"d": d, "x": x, "i_weights": i_weights, **({"ps": ps} if weighted else {})}
    if regression in (ols_panel, wols_panel):
        spec["delta_y"] = y
    else:
        spec.update(y=y, post=post, pre=True, treat=False)

    result = regression(**spec)

    np.testing.assert_allclose(result.coefficients, expected, rtol=1e-10)
    np.testing.assert_allclose(result.out_reg, x @ expected, rtol=1e-10)


@pytest.mark.parametrize("regression", [ols_panel, wols_panel, ols_rc, wols_rc])
def test_regressions_warn_about_a_nearly_collinear_design(regression_specs, regression):
    spec = regression_specs[regression.__name__]
    x = spec["x"]
    spec = {**spec, "x": np.column_stack([x[:, :2], x[:, 1] + 1e-11 * x[:, 2]])}

    with pytest.warns(UserWarning, match="^Potential multicollinearity detected \\(condition number: "):
        regression(**spec)


def test_collinearity_check_ignores_the_scale_of_a_covariate():
    rng = np.random.default_rng(3)
    z = rng.normal(size=1000)
    weights = rng.uniform(0.5, 1.5, size=1000)

    assert not _has_collinear_columns(np.column_stack([np.ones(1000), 1e13 * z]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), z, np.ones(1000)]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), z, 2 * z]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), np.zeros(1000)]), weights)


@pytest.mark.parametrize("regression", [ols_panel, wols_panel, ols_rc, wols_rc])
def test_regressions_reject_a_covariate_that_varies_only_on_rows_with_zero_weight(regression):
    post = np.repeat([0, 1, 0, 1], 6)
    d = np.repeat([0, 0, 1, 1], 6)
    rng = np.random.default_rng(15)
    x = np.column_stack([np.ones(24), rng.standard_normal(24), np.r_[np.zeros(4), 1.0, 2.0, np.zeros(18)]])
    i_weights = np.r_[np.ones(4), 0.0, 0.0, np.ones(18)]
    y = rng.standard_normal(24)
    extra = {"ps": np.full(24, 0.5)} if regression in (wols_panel, wols_rc) else {}
    if regression in (ols_panel, wols_panel):
        spec = {"delta_y": y, **extra}
        message = (
            "The regression design matrix of the comparison group is singular. Consider removing covariates that "
            "are constant or collinear among the comparison units."
        )
    else:
        spec = {"y": y, "post": post, "pre": True, "treat": False, **extra}
        message = (
            "The regression design matrix of the comparison group in the pre-period is singular. Consider removing "
            "covariates that are constant or collinear within that group and period."
        )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(d=d, x=x, i_weights=i_weights, **spec)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize(
    ("outcome", "covariate"),
    [(np.nan, 0.5), (np.inf, 0.5), (0.5, np.inf)],
    ids=["nan-outcome", "inf-outcome", "inf-covariate"],
)
def test_rc_regressions_reject_a_fit_that_cannot_be_solved(regression, outcome, covariate):
    post = np.repeat([0, 1, 0, 1], 4)
    d = np.repeat([0, 0, 1, 1], 4)
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(16), rng.standard_normal((16, 2))])
    y = rng.standard_normal(16)
    y[8] = outcome
    x[8, 1] = covariate
    extra = {"ps": np.full(16, 0.5)} if regression is wols_rc else {}
    message = (
        "The regression of the treated group in the pre-period could not be solved. Pass finite outcomes and "
        "covariates for that group and period."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(y=y, post=post, d=d, x=x, i_weights=np.ones(16), pre=True, treat=True, **extra)


@pytest.mark.parametrize("regression", [ols_panel, wols_panel])
def test_panel_regressions_reject_a_comparison_outcome_that_is_not_finite(regression):
    d = np.repeat([1, 0], [6, 10])
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(16), rng.standard_normal((16, 2))])
    delta_y = rng.standard_normal(16)
    delta_y[8] = np.nan
    extra = {"ps": np.full(16, 0.5)} if regression is wols_panel else {}

    with pytest.raises(ValueError, match="^Failed to solve linear system\\. Coefficients contain NaN/Inf values"):
        regression(delta_y=delta_y, d=d, x=x, i_weights=np.ones(16), **extra)


@pytest.mark.parametrize("regression", [ols_panel, wols_panel])
def test_panel_regressions_reject_collinear_covariates(regression):
    d = np.repeat([1, 0], [6, 10])
    z = np.random.default_rng(10).standard_normal(16)
    x = np.column_stack([np.ones(16), z, 3 * z])
    extra = {"ps": np.full(16, 0.5)} if regression is wols_panel else {}
    message = (
        "The regression design matrix of the comparison group is singular. Consider removing covariates that are "
        "constant or collinear among the comparison units."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        regression(delta_y=z + 1, d=d, x=x, i_weights=np.ones(16), **extra)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize("scale", [1e9, 1e16])
def test_rc_regressions_fit_a_covariate_on_any_scale(regression, scale):
    post = np.repeat([0, 1, 0, 1], 50)
    d = np.repeat([0, 0, 1, 1], 50)
    rng = np.random.default_rng(11)
    z = rng.standard_normal((200, 2))
    y = 1 + z @ np.array([2.0, -1.0]) + rng.standard_normal(200)
    extra = {"ps": np.full(200, 0.5)} if regression is wols_rc else {}
    spec = {"y": y, "post": post, "d": d, "i_weights": np.ones(200), "pre": True, "treat": False, **extra}

    expected = regression(x=np.column_stack([np.ones(200), z]), **spec)
    result = regression(x=np.column_stack([np.ones(200), z[:, 0], scale * z[:, 1]]), **spec)

    np.testing.assert_allclose(result.out_reg, expected.out_reg, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result.coefficients * [1, 1, scale], expected.coefficients, rtol=1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("regression", [ols_panel, wols_panel, ols_rc, wols_rc])
def test_regressions_of_host_arrays_under_the_cupy_backend_match_the_numpy_backend(regression_specs, regression):
    spec = regression_specs[regression.__name__]

    expected = regression(**spec)
    with use_backend("cupy"):
        result = regression(**spec)

    assert isinstance(result.out_reg, np.ndarray)
    np.testing.assert_allclose(
        result.out_reg, expected.out_reg, rtol=1e-10, atol=1e-10 * np.max(np.abs(expected.out_reg))
    )
    np.testing.assert_allclose(result.coefficients, expected.coefficients, rtol=1e-10)


@pytest.mark.gpu
@pytest.mark.parametrize("regression", [ols_panel, wols_panel, ols_rc, wols_rc])
def test_regressions_on_gpu_warn_about_a_nearly_collinear_design_like_cpu(cp, regression_specs, regression):
    spec = regression_specs[regression.__name__]
    x = spec["x"]
    spec = {**spec, "x": np.column_stack([x[:, :2], x[:, 1] + 1e-11 * x[:, 2]])}
    device_spec = {name: cp.asarray(value) if isinstance(value, np.ndarray) else value for name, value in spec.items()}

    with pytest.warns(UserWarning, match="Potential multicollinearity detected"):
        regression(**spec)
    with use_backend("cupy"), pytest.warns(UserWarning, match="Potential multicollinearity detected"):
        regression(**device_spec)
