"""Tests for the estimators."""

import warnings

import numpy as np
import pytest

from drsynthdid.estimators import aipw_did_panel


def assert_allclose_with_nans(actual, desired, rtol=1e-7, atol=1e-9, msg=""):
    if np.isnan(desired):
        assert np.isnan(actual), f"{msg}Expected NaN, got {actual}"
    else:
        assert not np.isnan(actual), f"{msg}Expected {desired}, got NaN"
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=msg)


def test_happy_path_unit_weights():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0, 3.0, 2.0])
    d = np.array([1, 1, 0, 0, 1, 0])
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.65, 0.35])
    out_reg = np.array([0.5, 0.6, 0.2, 0.3, 0.55, 0.25])
    i_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    expected_att = 0.6400224215246636
    actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
    assert_allclose_with_nans(actual_att, expected_att, rtol=1e-7, atol=1e-7)


def test_happy_path_non_uniform_weights():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0])
    d = np.array([1, 0, 1, 0])
    ps = np.array([0.6, 0.3, 0.7, 0.4])
    out_reg = np.array([0.5, 0.2, 0.6, 0.3])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2])

    expected_att = -1.367022086824067
    actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
    assert_allclose_with_nans(actual_att, expected_att, rtol=1e-7, atol=1e-7)


def test_all_treated():
    delta_y = np.array([1.0, 2.0, 3.0])
    d = np.array([1, 1, 1])
    ps = np.array([0.6, 0.7, 0.8])
    out_reg = np.array([0.5, 0.6, 0.7])
    i_weights = np.array([1.0, 1.0, 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
        assert len(w) >= 1
        assert any("Sum of w_cont is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_all_control():
    delta_y = np.array([1.0, 2.0, 3.0])
    d = np.array([0, 0, 0])
    ps = np.array([0.6, 0.7, 0.8])
    out_reg = np.array([0.5, 0.6, 0.7])
    i_weights = np.array([1.0, 1.0, 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
        assert len(w) >= 1
        assert any("Sum of w_treat is zero" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_ps_one_for_control_unit():
    delta_y = np.array([1.0, 0.0])
    d = np.array([1, 0])
    ps = np.array([0.5, 1.0])
    out_reg = np.array([0.5, 0.2])
    i_weights = np.array([1.0, 1.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
        assert len(w) >= 1
        assert any("Propensity score is 1 for some control units" in str(warn.message) for warn in w)
        assert any(
            "Sum of w_cont is inf" in str(warn.message) or "Sum of w_cont is nan" in str(warn.message) for warn in w
        )
    assert_allclose_with_nans(actual_att, np.nan)


def test_ps_zero_for_control_unit():
    delta_y = np.array([1.0, 2.0, 0.0])
    d = np.array([1, 0, 0])
    ps = np.array([0.7, 0.0, 0.2])
    out_reg = np.array([0.5, 0.1, 0.1])
    i_weights = np.array([1.0, 1.0, 1.0])

    expected_att = 0.6
    actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
    assert_allclose_with_nans(actual_att, expected_att)


def test_all_zero_i_weights():
    delta_y = np.array([1.0, 2.0, 0.0])
    d = np.array([1, 0, 1])
    ps = np.array([0.6, 0.3, 0.7])
    out_reg = np.array([0.5, 0.2, 0.6])
    i_weights = np.array([0.0, 0.0, 0.0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
        assert len(w) >= 1
        assert any("Mean of i_weights is zero" in str(warn.message) for warn in w)
        assert any("Sum of w_treat is zero" in str(warn.message) for warn in w)
        assert any("Sum of w_cont is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_some_zero_i_weights():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0])
    d = np.array([1, 0, 1, 0])
    ps = np.array([0.6, 0.3, 0.7, 0.4])
    out_reg = np.array([0.5, 0.2, 0.6, 0.3])
    i_weights = np.array([1.0, 0.0, 1.0, 1.0])

    expected_att = -0.75
    actual_att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)
    assert_allclose_with_nans(actual_att, expected_att, rtol=1e-5, atol=1e-5)


def test_input_type_errors():
    valid_np = np.array([1.0])
    invalid_list = [1.0]
    with pytest.raises(TypeError, match="All inputs .* must be NumPy arrays."):
        aipw_did_panel(invalid_list, valid_np, valid_np, valid_np, valid_np)
    with pytest.raises(TypeError, match="All inputs .* must be NumPy arrays."):
        aipw_did_panel(valid_np, invalid_list, valid_np, valid_np, valid_np)
    with pytest.raises(TypeError, match="All inputs .* must be NumPy arrays."):
        aipw_did_panel(valid_np, valid_np, invalid_list, valid_np, valid_np)
    with pytest.raises(TypeError, match="All inputs .* must be NumPy arrays."):
        aipw_did_panel(valid_np, valid_np, valid_np, invalid_list, valid_np)
    with pytest.raises(TypeError, match="All inputs .* must be NumPy arrays."):
        aipw_did_panel(valid_np, valid_np, valid_np, valid_np, invalid_list)


def test_input_shape_mismatch_error():
    delta_y = np.array([1.0, 2.0])
    d = np.array([1])
    ps = np.array([0.5, 0.5])
    out_reg = np.array([0.1, 0.1])
    i_weights = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="All input arrays must have the same shape."):
        aipw_did_panel(delta_y, d, ps, out_reg, i_weights)


def test_input_ndim_error():
    delta_y_2d = np.array([[1.0], [2.0]])
    d_1d = np.array([1, 0])
    with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
        aipw_did_panel(delta_y_2d, d_1d, d_1d, d_1d, d_1d)
    with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
        aipw_did_panel(d_1d, delta_y_2d, d_1d, d_1d, d_1d)


def test_empty_inputs():
    empty_arr = np.array([])
    with pytest.raises(ValueError, match="All input arrays must have the same shape."):
        aipw_did_panel(empty_arr, empty_arr, empty_arr, empty_arr, np.array([1.0]))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_panel(empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
        assert any("Sum of w_treat is zero" in str(warn.message) for warn in w)
        assert any("Sum of w_cont is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_non_finite_mean_i_weights():
    delta_y = np.array([1.0, 2.0])
    d = np.array([1, 0])
    ps = np.array([0.6, 0.4])
    out_reg = np.array([0.5, 0.3])
    i_weights_inf = np.array([1.0, np.inf])
    i_weights_nan = np.array([1.0, np.nan])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_inf = aipw_did_panel(delta_y, d, ps, out_reg, i_weights_inf)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_inf, np.nan)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_nan = aipw_did_panel(delta_y, d, ps, out_reg, i_weights_nan)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_nan, np.nan)
