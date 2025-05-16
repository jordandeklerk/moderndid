"""Tests for propensity-weighted estimators."""

import warnings

import numpy as np
import pytest

from drsynthdid.estimators import aipw_did_panel, aipw_did_rc

from .dgp import SantAnnaZhaoDRDiD

Y_RC_VALID = np.array([10.0, 12.0, 11.0, 13.0, 20.0, 22.0, 15.0, 18.0, 19.0, 25.0])
POST_RC_VALID = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)
D_RC_VALID = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=int)
PS_RC_VALID = np.array([0.4, 0.45, 0.38, 0.42, 0.6, 0.65, 0.58, 0.62, 0.55, 0.68])
OUT_Y_TREAT_POST_RC_VALID = np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 20.0, 26.0])
OUT_Y_TREAT_PRE_RC_VALID = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 10.0, 11.0])
OUT_Y_CONT_POST_RC_VALID = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 13.0, 14.0])
OUT_Y_CONT_PRE_RC_VALID = np.array([9.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0, 13.0, 9.0, 10.0])
I_WEIGHTS_RC_UNIT_VALID = np.ones(10)
I_WEIGHTS_RC_NON_UNIT_VALID = np.array([0.5, 1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.0, 1.4])

ALL_VALID_ARGS_RC = (
    Y_RC_VALID,
    POST_RC_VALID,
    D_RC_VALID,
    PS_RC_VALID,
    OUT_Y_TREAT_POST_RC_VALID,
    OUT_Y_TREAT_PRE_RC_VALID,
    OUT_Y_CONT_POST_RC_VALID,
    OUT_Y_CONT_PRE_RC_VALID,
    I_WEIGHTS_RC_UNIT_VALID,
)


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


def test_aipw_rc_happy_path_unit_weights():
    dgp = SantAnnaZhaoDRDiD(n_units=5000, treatment_fraction=0.5, common_support_strength=0.75, random_seed=123)
    att_val = 0.5
    data = dgp.generate_data(att=att_val)

    y_arr = data["df"]["y"].to_numpy()
    post_arr = data["df"]["post"].to_numpy()
    d_arr = data["df"]["d"].to_numpy()
    ps_arr = data["propensity_scores"]

    out_y_treat_pre_arr = data["potential_outcomes_pre"]["y0"]
    out_y_cont_pre_arr = data["potential_outcomes_pre"]["y0"]
    out_y_cont_post_arr = data["potential_outcomes_post"]["y0"]
    out_y_treat_post_arr = data["potential_outcomes_post"]["y1"]

    i_weights_arr = np.ones_like(y_arr)

    actual_att = aipw_did_rc(
        y_arr,
        post_arr,
        d_arr,
        ps_arr,
        out_y_treat_post_arr,
        out_y_treat_pre_arr,
        out_y_cont_post_arr,
        out_y_cont_pre_arr,
        i_weights_arr,
    )
    assert_allclose_with_nans(actual_att, data["true_att"], rtol=5e-2, atol=5e-2)


def test_aipw_rc_happy_path_non_uniform_weights():
    dgp = SantAnnaZhaoDRDiD(n_units=5000, treatment_fraction=0.5, common_support_strength=0.75, random_seed=456)
    att_val = 0.3
    data = dgp.generate_data(att=att_val)

    y_arr = data["df"]["y"].to_numpy()
    post_arr = data["df"]["post"].to_numpy()
    d_arr = data["df"]["d"].to_numpy()
    ps_arr = data["propensity_scores"]

    out_y_treat_pre_arr = data["potential_outcomes_pre"]["y0"]
    out_y_cont_pre_arr = data["potential_outcomes_pre"]["y0"]
    out_y_cont_post_arr = data["potential_outcomes_post"]["y0"]
    out_y_treat_post_arr = data["potential_outcomes_post"]["y1"]

    rng = np.random.RandomState(789)
    i_weights_arr = rng.rand(dgp.n_units) + 0.5

    actual_att = aipw_did_rc(
        y_arr,
        post_arr,
        d_arr,
        ps_arr,
        out_y_treat_post_arr,
        out_y_treat_pre_arr,
        out_y_cont_post_arr,
        out_y_cont_pre_arr,
        i_weights_arr,
    )
    assert_allclose_with_nans(actual_att, data["true_att"], rtol=5e-2, atol=5e-2)


def test_aipw_rc_ps_one_for_control_unit():
    args = list(ALL_VALID_ARGS_RC)
    ps_mod = PS_RC_VALID.copy()
    ps_mod[0] = 1.0
    ps_mod[2] = 1.0
    args[3] = ps_mod

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*args)
        assert any("Propensity score is 1 for some control units" in str(warn.message) for warn in w)
        assert any(
            name in str(warn.message).lower()
            for warn in w
            for name in ["att_cont_pre is inf", "att_cont_pre is nan", "att_cont_post is inf", "att_cont_post is nan"]
        )
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_all_zero_i_weights():
    args = list(ALL_VALID_ARGS_RC)
    args[-1] = np.zeros_like(I_WEIGHTS_RC_UNIT_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*args)
        assert any("Mean of i_weights is zero" in str(warn.message) for warn in w)
        term_names = ["att_treat_pre", "att_treat_post", "att_d_post", "att_dt1_post", "att_d_pre", "att_dt0_pre"]
        for term_name in term_names:
            assert any(f"Sum of weights for {term_name} is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_non_finite_mean_i_weights():
    args_inf = list(ALL_VALID_ARGS_RC)
    i_weights_inf = I_WEIGHTS_RC_UNIT_VALID.copy()
    i_weights_inf[0] = np.inf
    args_inf[-1] = i_weights_inf

    args_nan = list(ALL_VALID_ARGS_RC)
    i_weights_nan = I_WEIGHTS_RC_UNIT_VALID.copy()
    i_weights_nan[0] = np.nan
    args_nan[-1] = i_weights_nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_inf = aipw_did_rc(*args_inf)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_inf, np.nan)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_nan = aipw_did_rc(*args_nan)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_nan, np.nan)


@pytest.mark.parametrize("invalid_arg_idx", range(len(ALL_VALID_ARGS_RC)))
def test_aipw_rc_input_type_errors(invalid_arg_idx):
    args = list(ALL_VALID_ARGS_RC)
    args[invalid_arg_idx] = list(args[invalid_arg_idx])
    with pytest.raises(TypeError, match="All inputs must be NumPy arrays."):
        aipw_did_rc(*args)


@pytest.mark.parametrize("mismatch_arg_idx", range(len(ALL_VALID_ARGS_RC)))
def test_aipw_rc_input_shape_mismatch_error(mismatch_arg_idx):
    args = list(ALL_VALID_ARGS_RC)
    original_shape_len = args[mismatch_arg_idx].shape[0]
    if original_shape_len > 1:
        short_arr = np.array([1.0] * (original_shape_len - 1))
        args[mismatch_arg_idx] = short_arr
        with pytest.raises(ValueError, match="All input arrays must have the same shape."):
            aipw_did_rc(*args)
    elif original_shape_len == 1:
        long_arr = np.array([1.0] * (original_shape_len + 1))
        args[mismatch_arg_idx] = long_arr
        with pytest.raises(ValueError, match="All input arrays must have the same shape."):
            aipw_did_rc(*args)


@pytest.mark.parametrize("ndim_arg_idx", range(len(ALL_VALID_ARGS_RC)))
def test_aipw_rc_input_ndim_error(ndim_arg_idx):
    args = list(ALL_VALID_ARGS_RC)
    arr_2d = np.array([args[ndim_arg_idx]])
    if args[ndim_arg_idx].size > 0:
        args[ndim_arg_idx] = arr_2d
        with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
            aipw_did_rc(*args)
    elif arr_2d.shape == (1, 0) and args[ndim_arg_idx].shape == (0,):
        args[ndim_arg_idx] = arr_2d
        with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
            aipw_did_rc(*args)


def test_aipw_rc_empty_inputs():
    empty_arr = np.array([])
    empty_args = [empty_arr] * len(ALL_VALID_ARGS_RC)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*empty_args)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
        term_names = [
            "att_treat_pre",
            "att_treat_post",
            "att_cont_pre",
            "att_cont_post",
            "att_d_post",
            "att_dt1_post",
            "att_d_pre",
            "att_dt0_pre",
        ]
        for term_name in term_names:
            assert any(f"Sum of weights for {term_name} is 0.0" in str(warn.message) for warn in w) or any(
                f"Sum of weights for {term_name} is nan" in str(warn.message) for warn in w
            )
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_specific_term_nan_due_to_zero_weights():
    args = list(ALL_VALID_ARGS_RC)
    args[1] = np.ones_like(POST_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*args)
        assert any("Sum of weights for att_treat_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_cont_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_dt0_pre is 0.0" in str(warn.message) for warn in w)

    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_no_treated_units():
    args = list(ALL_VALID_ARGS_RC)
    args[2] = np.zeros_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*args)
        assert any("Sum of weights for att_treat_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_treat_post is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_d_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_no_control_units():
    args = list(ALL_VALID_ARGS_RC)
    args[2] = np.ones_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc(*args)
        assert any("Sum of weights for att_cont_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_cont_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)
