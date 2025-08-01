"""Tests for weighted OLS regression functions."""

import warnings

import numpy as np
import pytest

from causaldid import wols_panel, wols_rc


def test_wols_panel_happy_path_unit_weights():
    n_units = 10
    n_features = 3

    delta_y = np.array([1.5, 2.0, 0.5, 1.0, 0.8, 1.2, 0.3, 0.9, 1.1, 0.7])
    d = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=int)
    x = np.random.RandomState(42).randn(n_units, n_features)
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
    x = np.random.RandomState(123).randn(n_units, n_features)
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
    x = np.random.RandomState(456).randn(n_units, n_features)
    ps = np.array([0.6, 0.7, 0.65, 0.8, 0.75])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="No control units found"):
        wols_panel(delta_y, d, x, ps, i_weights)


def test_wols_panel_few_control_units():
    n_units = 10
    n_features = 2

    delta_y = np.array([1.0, 2.0, 1.5, 1.2, 0.8, 0.5, 0.9, 1.1, 0.7, 0.6])
    d = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=int)
    x = np.random.RandomState(789).randn(n_units, n_features)
    ps = np.array([0.6, 0.7, 0.65, 0.8, 0.75, 0.72, 0.68, 0.3, 0.25, 0.35])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_panel(delta_y, d, x, ps, i_weights)
        assert len(w) >= 1
        assert any("Only 3 control units available" in str(warn.message) for warn in w)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")


def test_wols_panel_ps_one_for_control():
    n_units = 5
    n_features = 2

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8])
    d = np.array([1, 0, 0, 1, 0], dtype=int)
    x = np.random.RandomState(111).randn(n_units, n_features)
    ps = np.array([0.6, 1.0, 0.4, 0.7, 0.35])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="Propensity score is 1 for some control units"):
        wols_panel(delta_y, d, x, ps, i_weights)


def test_wols_panel_extreme_weight_ratios():
    n_units = 6
    n_features = 2

    delta_y = np.array([1.0, 2.0, 0.5, 1.5, 0.8, 1.2])
    d = np.array([1, 0, 0, 0, 1, 0], dtype=int)
    x = np.random.RandomState(222).randn(n_units, n_features)
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
    x[:, 0] = np.random.RandomState(333).randn(n_units)
    x[:, 1] = x[:, 0] * 2  # Ensure perfect multicollinearity
    x[:, 2] = x[:, 0] + x[:, 1]
    ps = np.array([0.6, 0.3, 0.4, 0.35, 0.7, 0.25, 0.45, 0.38, 0.65, 0.32])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="Failed to solve linear system"):
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


def test_wols_rc_happy_path_control_post():
    n_units = 10
    n_features = 3

    y = np.array([10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 15.0, 18.0, 19.0, 25.0])
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    x = np.random.RandomState(444).randn(n_units, n_features)
    ps = np.array([0.4, 0.45, 0.38, 0.42, 0.6, 0.65, 0.58, 0.62, 0.55, 0.68])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_rc(y, post, d, x, ps, i_weights, pre=False, treat=False)
        assert len(w) >= 1

        assert any(
            "Number of observations in subset (2) is less than the number of features (3)" in str(warn.message)
            for warn in w
        )

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
    assert np.all(np.isnan(result.out_reg))
    assert np.all(np.isnan(result.coefficients))


def test_wols_rc_happy_path_treat_pre():
    n_units = 8
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    x = np.random.RandomState(555).randn(n_units, n_features)
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
    x = np.random.RandomState(666).randn(n_units, n_features)
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
    x = np.random.RandomState(777).randn(n_units, n_features)
    ps = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.42])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="No units found for pre=True, treat=False"):
        wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)


def test_wols_rc_few_units_in_subset():
    n_units = 10
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0, 19.0, 21.0])
    post = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    x = np.random.RandomState(888).randn(n_units, n_features)
    ps = np.array([0.4, 0.5, 0.3, 0.45, 0.35, 0.6, 0.7, 0.65, 0.72, 0.68])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)
        assert len(w) >= 1

        assert any(
            "Number of observations in subset (1) is less than the number of features (2)" in str(warn.message)
            for warn in w
        )

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
    assert np.all(np.isnan(result.out_reg))
    assert np.all(np.isnan(result.coefficients))


def test_wols_rc_ps_one_in_subset():
    n_units = 6
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0])
    post = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1], dtype=int)
    x = np.random.RandomState(999).randn(n_units, n_features)
    ps = np.array([1.0, 0.5, 0.4, 1.0, 0.6, 0.7])
    i_weights = np.ones(n_units)

    with pytest.raises(ValueError, match="Propensity score is 1 for some units in subset"):
        wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)


def test_wols_rc_extreme_weights():
    n_units = 8
    n_features = 2

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0])
    post = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    x = np.random.RandomState(1111).randn(n_units, n_features)
    ps = np.array([0.001, 0.5, 0.4, 0.999, 0.6, 0.7, 0.65, 0.72])
    i_weights = np.ones(n_units)

    result = wols_rc(y, post, d, x, ps, i_weights, pre=False, treat=False)

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")


def test_wols_rc_multicollinearity():
    n_units = 10
    n_features = 3

    y = np.array([10.0, 15.0, 12.0, 18.0, 20.0, 22.0, 16.0, 14.0, 19.0, 21.0])
    post = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    x = np.zeros((n_units, n_features))
    x[:, 0] = np.random.RandomState(2222).randn(n_units)
    x[:, 1] = x[:, 0] * 2
    x[:, 2] = x[:, 0] + x[:, 1]
    ps = np.array([0.4, 0.45, 0.38, 0.42, 0.6, 0.65, 0.58, 0.62, 0.55, 0.68])
    i_weights = np.ones(n_units)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = wols_rc(y, post, d, x, ps, i_weights, pre=True, treat=False)
        assert len(w) >= 1
        assert any(
            "Number of observations in subset (2) is less than the number of features (3)" in str(warn.message)
            for warn in w
        )

    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert np.all(np.isnan(result.out_reg))
    assert np.all(np.isnan(result.coefficients))


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

    y = np.random.RandomState(3333).randn(n_units) + 15
    post = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=int)
    d = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
    x = np.random.RandomState(3334).randn(n_units, n_features)
    ps = np.array([0.3, 0.35, 0.32, 0.38, 0.4, 0.36, 0.6, 0.65, 0.62, 0.68, 0.7, 0.66])
    i_weights = np.ones(n_units)

    result = wols_rc(y, post, d, x, ps, i_weights, pre=pre, treat=treat)
    assert hasattr(result, "out_reg")
    assert hasattr(result, "coefficients")
    assert result.out_reg.shape == (n_units,)
    assert result.coefficients.shape == (n_features,)
