"""Tests for weighted OLS regression functions."""

import re
import warnings

import numpy as np
import pytest

from moderndid import ols_panel, wols_panel, wols_rc
from moderndid.drdid.estimators.wols import _has_collinear_columns, ols_rc


def test_ols_panel_happy_path():
    n_units = 10
    n_features = 3

    rng = np.random.default_rng(42)
    delta_y = np.array([1.5, 2.0, 0.5, 1.0, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = rng.standard_normal((n_units, n_features))
    i_weights = np.ones(n_units)

    result = ols_panel(delta_y, d, x, i_weights)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
    assert np.all(np.isfinite(result.out_reg))
    assert np.all(np.isfinite(result.coefficients))


def test_wols_panel_happy_path_unit_weights():
    n_units = 10
    n_features = 3

    delta_y = np.array([1.5, 2.0, 0.5, 1.0, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(42).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.35, 0.65, 0.25, 0.45, 0.55, 0.38])
    i_weights = np.ones(n_units)

    result = wols_panel(delta_y, d, x, ps, i_weights)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
    assert np.all(np.isfinite(result.out_reg))
    assert np.all(np.isfinite(result.coefficients))


def test_wols_panel_happy_path_non_uniform_weights():
    n_units = 8
    n_features = 2

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2, 0.3, 0.9])
    d = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(123).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.3, 0.4, 0.7, 0.35, 0.25, 0.55, 0.45])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7])

    result = wols_panel(delta_y, d, x, ps, i_weights)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
    assert np.all(np.isfinite(result.out_reg))
    assert np.all(np.isfinite(result.coefficients))


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
    n_units = 10
    n_features = 2

    delta_y = np.array([1.0, 2.0, 1.5, 1.2, 0.8, 0.5, 0.9, 1.1, 0.7, 0.6])
    d = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=int)
    x = np.random.default_rng(789).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.7, 0.65, 0.8, 0.75, 0.72, 0.68, 0.3, 0.25, 0.35])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_panel(delta_y, d, x, ps, i_weights)
        assert len(w) >= 1
        assert any("Only 3 control units available" in str(warn.message) for warn in w)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")


@pytest.mark.filterwarnings("ignore:Only.*control units available:UserWarning")
def test_wols_panel_ps_one_for_control():
    n_units = 5
    n_features = 2

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8])
    d = np.array([1, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(111).standard_normal((n_units, n_features))
    ps = np.array([0.6, 1.0, 0.4, 0.7, 0.35])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="Propensity score is 1 for some control units"):
        wols_panel(delta_y, d, x, ps, i_weights)


def test_wols_panel_extreme_weight_ratios():
    n_units = 6
    n_features = 2

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2])
    d = np.array([1, 0, 0, 0, 1, 0], dtype=int)
    x = np.random.default_rng(222).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.999999, 0.001, 0.4, 0.7, 0.35])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_panel(delta_y, d, x, ps, i_weights)
        assert len(w) >= 1
        assert any("Extreme weight ratios detected" in str(warn.message) for warn in w)

    assert hasattr(result, "out_reg")


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


def test_wols_rc_happy_path_treat_pre():
    n_units = 8
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    x = np.random.default_rng(555).standard_normal((n_units, n_features))
    ps = np.array([0.6, 0.4, 0.65, 0.35, 0.55, 0.45, 0.58, 0.42])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2, 0.9, 1.1, 1.3, 0.7])

    result = wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=True)

    assert hasattr(result, "out_reg")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)


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


@pytest.mark.filterwarnings("ignore:Only.*units available:UserWarning")
def test_wols_rc_ps_one_in_subset():
    n_units = 6
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0])
    post = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    x = np.random.default_rng(999).standard_normal((n_units, n_features))
    ps = np.array([1.0, 0.5, 0.4, 1.0, 0.6, 0.7])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="Propensity score is 1 for some units in subset"):
        wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)


@pytest.mark.filterwarnings("ignore:Only.*units available:UserWarning")
def test_wols_rc_extreme_weights():
    n_units = 8
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    x = np.random.default_rng(1111).standard_normal((n_units, n_features))
    ps = np.array([0.001, 0.5, 0.4, 0.999, 0.6, 0.7, 0.65, 0.72])
    i_weights = np.ones(n_units)

    result = wols_rc(y, post, d, x, ps, i_weights, pre=False, treat=False)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")


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


@pytest.mark.parametrize("pre,treat", [(True, True), (True, False), (False, True), (False, False)])
def test_wols_rc_all_subset_combinations(pre, treat):
    n_units = 12
    n_features = 2

    y = np.random.default_rng(3333).standard_normal(n_units) + 15
    post = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    x = np.random.default_rng(3334).standard_normal((n_units, n_features))
    ps = np.array([0.3, 0.35, 0.32, 0.38, 0.4, 0.36, 0.6, 0.65, 0.62, 0.68, 0.7, 0.66])
    i_weights = np.ones(n_units)

    result = wols_rc(y, post, d, x, ps, i_weights, pre=pre, treat=treat)
    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)


def test_collinearity_check_ignores_the_scale_of_a_covariate():
    rng = np.random.default_rng(3)
    z = rng.normal(size=1000)
    weights = rng.uniform(0.5, 1.5, size=1000)

    assert not _has_collinear_columns(np.column_stack([np.ones(1000), 1e13 * z]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), z, np.ones(1000)]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), z, 2 * z]), weights)
    assert _has_collinear_columns(np.column_stack([np.ones(1000), np.zeros(1000)]), weights)


@pytest.mark.parametrize("regression", [wols_rc, ols_rc])
@pytest.mark.parametrize(("outcome", "covariate"), [(np.nan, 0.5), (0.5, np.inf)], ids=["nan-outcome", "inf-covariate"])
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
    np.testing.assert_allclose(result.coefficients[2] * scale, expected.coefficients[2], rtol=1e-12)
