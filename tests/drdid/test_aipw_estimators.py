"""Tests for the augmented inverse propensity weighted DiD estimators."""

import re
import warnings

import numpy as np
import pytest

from moderndid import aipw_did_panel, aipw_did_rc_imp1, aipw_did_rc_imp2


def test_happy_path_unit_weights(aipw_panel_arrays):
    treated = (0.5 + 1.4 + 2.45) / 3
    control = (3 / 7 * -0.2 + 2 / 3 * 0.7 + 7 / 13 * 1.75) / (3 / 7 + 2 / 3 + 7 / 13)

    att = aipw_did_panel(**aipw_panel_arrays)

    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_happy_path_non_uniform_weights():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0])
    d = np.array([1, 0, 1, 0])
    ps = np.array([0.6, 0.3, 0.7, 0.4])
    out_reg = np.array([0.5, 0.2, 0.6, 0.3])
    i_weights = np.array([0.5, 1.5, 0.8, 1.2])
    treated = (0.5 * 0.5 + 0.8 * -0.6) / (0.5 + 0.8)
    control = (1.5 * 3 / 7 * 1.8 + 1.2 * 2 / 3 * 0.7) / (1.5 * 3 / 7 + 1.2 * 2 / 3)

    att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_all_treated():
    delta_y = np.array([1.0, 2.0, 3.0])
    d = np.array([1, 1, 1])
    ps = np.array([0.6, 0.7, 0.8])
    out_reg = np.array([0.5, 0.6, 0.7])
    i_weights = np.array([1.0, 1.0, 1.0])

    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of w_cont is 0.0 (no effectively control units or problematic weights). aipw_0 will be NaN."
    ]
    assert np.isnan(att)


def test_all_control():
    delta_y = np.array([1.0, 2.0, 3.0])
    d = np.array([0, 0, 0])
    ps = np.array([0.6, 0.7, 0.8])
    out_reg = np.array([0.5, 0.6, 0.7])
    i_weights = np.array([1.0, 1.0, 1.0])

    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of w_treat is zero (no effectively treated units). aipw_1 will be NaN."
    ]
    assert np.isnan(att)


def test_ps_one_for_control_unit():
    delta_y = np.array([1.0, 0.0])
    d = np.array([1, 0])
    ps = np.array([0.5, 1.0])
    out_reg = np.array([0.5, 0.2])
    i_weights = np.array([1.0, 1.0])

    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Propensity score is 1 for some control units. Their weights will be NaN/Inf. "
        "This typically indicates issues with the propensity score model.",
        "Sum of w_cont is inf (no effectively control units or problematic weights). aipw_0 will be NaN.",
    ]
    assert np.isnan(att)


def test_ps_zero_for_control_unit():
    delta_y = np.array([1.0, 2.0, 0.0])
    d = np.array([1, 0, 0])
    ps = np.array([0.7, 0.0, 0.2])
    out_reg = np.array([0.5, 0.1, 0.1])
    i_weights = np.array([1.0, 1.0, 1.0])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record] == []
    np.testing.assert_allclose(att, (1.0 - 0.5) - (0.0 - 0.1), rtol=1e-12, atol=0)


def test_all_zero_i_weights():
    delta_y = np.array([1.0, 2.0, 0.0])
    d = np.array([1, 0, 1])
    ps = np.array([0.6, 0.3, 0.7])
    out_reg = np.array([0.5, 0.2, 0.6])
    i_weights = np.array([0.0, 0.0, 0.0])

    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is zero, cannot normalize. Using original weights.",
        "Sum of w_treat is zero (no effectively treated units). aipw_1 will be NaN.",
        "Sum of w_cont is 0.0 (no effectively control units or problematic weights). aipw_0 will be NaN.",
    ]
    assert np.isnan(att)


def test_some_zero_i_weights():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0])
    d = np.array([1, 0, 1, 0])
    ps = np.array([0.6, 0.3, 0.7, 0.4])
    out_reg = np.array([0.5, 0.2, 0.6, 0.3])
    i_weights = np.array([1.0, 0.0, 1.0, 1.0])

    att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    np.testing.assert_allclose(att, (0.5 + -0.6) / 2 - 0.7, rtol=1e-12, atol=0)


def test_input_type_errors():
    valid_np = np.array([1.0])
    invalid_list = [1.0]
    message = re.escape("All inputs must be NumPy arrays.")

    with pytest.raises(TypeError, match=message):
        aipw_did_panel(invalid_list, valid_np, valid_np, valid_np, valid_np)
    with pytest.raises(TypeError, match=message):
        aipw_did_panel(valid_np, invalid_list, valid_np, valid_np, valid_np)
    with pytest.raises(TypeError, match=message):
        aipw_did_panel(valid_np, valid_np, invalid_list, valid_np, valid_np)
    with pytest.raises(TypeError, match=message):
        aipw_did_panel(valid_np, valid_np, valid_np, invalid_list, valid_np)
    with pytest.raises(TypeError, match=message):
        aipw_did_panel(valid_np, valid_np, valid_np, valid_np, invalid_list)


def test_input_shape_mismatch_error():
    delta_y = np.array([1.0, 2.0])
    d = np.array([1])
    ps = np.array([0.5, 0.5])
    out_reg = np.array([0.1, 0.1])
    i_weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError, match=re.escape("All input arrays must have the same shape.")):
        aipw_did_panel(delta_y, d, ps, out_reg, i_weights)


def test_input_ndim_error():
    delta_y_2d = np.array([[1.0], [2.0]])
    d_1d = np.array([1, 0])
    message = re.escape("All input arrays must be 1-dimensional.")

    with pytest.raises(ValueError, match=message):
        aipw_did_panel(delta_y_2d, d_1d, d_1d, d_1d, d_1d)
    with pytest.raises(ValueError, match=message):
        aipw_did_panel(d_1d, delta_y_2d, d_1d, d_1d, d_1d)


def test_empty_inputs():
    empty = np.array([])

    with pytest.raises(ValueError, match=re.escape("All input arrays must have the same shape.")):
        aipw_did_panel(empty, empty, empty, empty, np.array([1.0]))
    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(empty, empty, empty, empty, empty)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is not finite. Using original weights.",
        "Sum of w_treat is zero (no effectively treated units). aipw_1 will be NaN.",
        "Sum of w_cont is 0.0 (no effectively control units or problematic weights). aipw_0 will be NaN.",
    ]
    assert np.isnan(att)


@pytest.mark.parametrize("weight, control_sum", [(np.inf, "inf"), (np.nan, "nan")])
def test_non_finite_mean_i_weights(weight, control_sum):
    delta_y = np.array([1.0, 2.0])
    d = np.array([1, 0])
    ps = np.array([0.6, 0.4])
    out_reg = np.array([0.5, 0.3])
    i_weights = np.array([1.0, weight])

    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(delta_y, d, ps, out_reg, i_weights)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is not finite. Using original weights.",
        f"Sum of w_cont is {control_sum} (no effectively control units or problematic weights). aipw_0 will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_happy_path_unit_weights(aipw_rc_imp2_arrays):
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)
    efficiency_post = (2 + 4 + 1 + 3 + 6 + 5) / 6 - (1 + 3 + 5) / 3
    efficiency_pre = (1 + 2 + 4 + 2 + 3 + 5) / 6 - (1 + 2 + 3) / 3

    att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays)

    np.testing.assert_allclose(att, treated - control + efficiency_post - efficiency_pre, rtol=1e-12, atol=0)


def test_aipw_rc_happy_path_non_uniform_weights(aipw_rc_imp2_arrays):
    i_weights = np.array([2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 2.0])
    treated = (1 * 7 + 3 * 4 + 2 * 10) / 6 - (1 * 3 + 2 * 5 + 3 * 1) / 6
    control = (3 * (1 / 3) * 1 + 1 * 4 * 6) / (3 * (1 / 3) + 1 * 4) - (2 * 1 * 2 + 1 * 3 * -2) / (2 * 1 + 1 * 3)
    efficiency_post = (1 * 2 + 2 * 4 + 1 * 1 + 3 * 3 + 3 * 6 + 2 * 5) / 12 - (1 * 1 + 3 * 3 + 2 * 5) / 6
    efficiency_pre = (1 * 1 + 2 * 2 + 1 * 4 + 3 * 2 + 3 * 3 + 2 * 5) / 12 - (1 * 1 + 2 * 2 + 3 * 3) / 6

    att = aipw_did_rc_imp2(**{**aipw_rc_imp2_arrays, "i_weights": i_weights})

    np.testing.assert_allclose(att, treated - control + efficiency_post - efficiency_pre, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "units, terms",
    [([0, 2], ["att_cont_pre", "att_cont_post"]), ([0], ["att_cont_pre"]), ([2], ["att_cont_post"])],
)
def test_aipw_rc_ps_one_for_control_unit(aipw_rc_imp2_arrays, units, terms):
    aipw_rc_imp2_arrays["ps"][units] = 1.0

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Propensity score is 1 for some control units. Their weights (w_cont_pre/w_cont_post) "
        "will be NaN/Inf. This typically indicates issues with the propensity score model.",
        *[f"Sum of weights for {term} is inf. Term will be NaN." for term in terms],
    ]
    assert np.isnan(att)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "unit, weight, warned",
    [
        (0, 1.0, "Sum of weights for att_cont_pre is inf. Term will be NaN."),
        (2, 1.0, "Sum of weights for att_cont_post is inf. Term will be NaN."),
        (0, -1.0, "Sum of weights for att_cont_pre is -inf. Term will be NaN."),
    ],
)
def test_aipw_rc_control_with_propensity_score_one_keeps_its_infinite_weight(
    aipw_rc_imp2_arrays, forbid_errstate, unit, weight, warned
):
    aipw_rc_imp2_arrays["ps"][unit] = 1.0
    aipw_rc_imp2_arrays["i_weights"][unit] = weight
    forbid_errstate()

    with pytest.warns(UserWarning) as record:
        actual_att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays)

    control_sums = [str(warn.message) for warn in record if str(warn.message).startswith("Sum of weights for att_cont")]
    assert control_sums == [warned]
    assert np.isnan(actual_att)


@pytest.mark.filterwarnings("error")
def test_aipw_rc_treated_unit_with_propensity_score_one(aipw_rc_imp2_arrays, forbid_errstate):
    aipw_rc_imp2_arrays["ps"][4] = 1.0
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)
    efficiency_post = (2 + 4 + 1 + 3 + 6 + 5) / 6 - (1 + 3 + 5) / 3
    efficiency_pre = (1 + 2 + 4 + 2 + 3 + 5) / 6 - (1 + 2 + 3) / 3
    forbid_errstate()

    att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays)

    np.testing.assert_allclose(att, treated - control + efficiency_post - efficiency_pre, rtol=1e-12, atol=0)


def test_aipw_rc_all_zero_i_weights(aipw_rc_imp2_arrays):
    terms = ["att_treat_pre", "att_treat_post", "att_cont_pre", "att_cont_post"]
    terms += ["att_d_post", "att_dt1_post", "att_d_pre", "att_dt0_pre"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**{**aipw_rc_imp2_arrays, "i_weights": np.zeros(10)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is zero, cannot normalize. Using original weights.",
        *[f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms],
    ]
    assert np.isnan(att)


@pytest.mark.parametrize("weight", [np.inf, np.nan])
def test_aipw_rc_non_finite_mean_i_weights(aipw_rc_imp2_arrays, weight):
    aipw_rc_imp2_arrays["i_weights"][0] = weight

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays)

    assert "Mean of i_weights is not finite. Using original weights." in [str(w.message) for w in record]
    assert np.isnan(att)


@pytest.mark.parametrize(
    "name",
    ["y", "post", "d", "ps", "out_y_treat_post", "out_y_treat_pre", "out_y_cont_post", "out_y_cont_pre", "i_weights"],
)
def test_aipw_rc_input_type_errors(aipw_rc_imp2_arrays, name):
    aipw_rc_imp2_arrays[name] = aipw_rc_imp2_arrays[name].tolist()

    with pytest.raises(TypeError, match=re.escape("All inputs must be NumPy arrays.")):
        aipw_did_rc_imp2(**aipw_rc_imp2_arrays)


@pytest.mark.parametrize(
    "name",
    ["y", "post", "d", "ps", "out_y_treat_post", "out_y_treat_pre", "out_y_cont_post", "out_y_cont_pre", "i_weights"],
)
def test_aipw_rc_input_shape_mismatch_error(aipw_rc_imp2_arrays, name):
    aipw_rc_imp2_arrays[name] = aipw_rc_imp2_arrays[name][:-1]

    with pytest.raises(ValueError, match=re.escape("All input arrays must have the same shape.")):
        aipw_did_rc_imp2(**aipw_rc_imp2_arrays)


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


ALL_VALID_ARGS_RC_BASIC = (
    Y_RC_VALID,
    POST_RC_VALID,
    D_RC_VALID,
    PS_RC_VALID,
    np.array([10.5, 11.8, 11.2, 12.5, 18.5, 20.2, 14.8, 16.5, 18.2, 23.0]),
    I_WEIGHTS_RC_UNIT_VALID,
)


def assert_allclose_with_nans(actual, desired, rtol=1e-7, atol=1e-9, msg=""):
    if np.isnan(desired):
        assert np.isnan(actual), f"{msg}Expected NaN, got {actual}"
    else:
        assert not np.isnan(actual), f"{msg}Expected {desired}, got NaN"
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=msg)


@pytest.mark.parametrize("ndim_arg_idx", range(len(ALL_VALID_ARGS_RC)))
def test_aipw_rc_input_ndim_error(ndim_arg_idx):
    args = list(ALL_VALID_ARGS_RC)
    arr_2d = np.array([args[ndim_arg_idx]])
    if args[ndim_arg_idx].size > 0:
        args[ndim_arg_idx] = arr_2d
        with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
            aipw_did_rc_imp2(*args)
    elif arr_2d.shape == (1, 0) and args[ndim_arg_idx].shape == (0,):
        args[ndim_arg_idx] = arr_2d
        with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
            aipw_did_rc_imp2(*args)


def test_aipw_rc_empty_inputs():
    empty_arr = np.array([])
    empty_args = [empty_arr] * len(ALL_VALID_ARGS_RC)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp2(*empty_args)
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
        actual_att = aipw_did_rc_imp2(*args)
        assert any("Sum of weights for att_treat_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_cont_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_dt0_pre is 0.0" in str(warn.message) for warn in w)

    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_no_treated_units():
    args = list(ALL_VALID_ARGS_RC)
    args[2] = np.zeros_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp2(*args)
        assert any("Sum of weights for att_treat_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_treat_post is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_d_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_no_control_units():
    args = list(ALL_VALID_ARGS_RC)
    args[2] = np.ones_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp2(*args)
        assert any("Sum of weights for att_cont_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for att_cont_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_happy_path_unit_weights():
    y, post, d, ps, out_reg, i_weights = ALL_VALID_ARGS_RC_BASIC

    actual_att = aipw_did_rc_imp1(y, post, d, ps, out_reg, i_weights)
    assert isinstance(actual_att, float)
    assert np.isfinite(actual_att)


def test_aipw_rc_basic_happy_path_non_uniform_weights():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[-1] = I_WEIGHTS_RC_NON_UNIT_VALID

    actual_att = aipw_did_rc_imp1(*args)
    assert isinstance(actual_att, float)
    assert np.isfinite(actual_att)


def test_aipw_rc_basic_ps_one_for_control_unit():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    ps_mod = PS_RC_VALID.copy()
    ps_mod[0] = 1.0
    ps_mod[2] = 1.0
    args[3] = ps_mod

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Propensity score is 1 for some control units" in str(warn.message) for warn in w)
        assert any(
            name in str(warn.message).lower()
            for warn in w
            for name in ["aipw_0_pre is inf", "aipw_0_pre is nan", "aipw_0_post is inf", "aipw_0_post is nan"]
        )
    assert_allclose_with_nans(actual_att, np.nan)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "unit, weight, warned",
    [
        (0, 1.0, "Sum of weights for aipw_0_pre is inf. Term will be NaN."),
        (2, 1.0, "Sum of weights for aipw_0_post is inf. Term will be NaN."),
        (0, -1.0, "Sum of weights for aipw_0_pre is -inf. Term will be NaN."),
    ],
)
def test_aipw_rc_basic_control_with_propensity_score_one_keeps_its_infinite_weight(
    forbid_errstate, unit, weight, warned
):
    args = [arr.copy() for arr in ALL_VALID_ARGS_RC_BASIC]
    args[3][unit] = 1.0
    args[-1][unit] = weight
    forbid_errstate()

    with pytest.warns(UserWarning) as record:
        actual_att = aipw_did_rc_imp1(*args)

    control_sums = [str(warn.message) for warn in record if str(warn.message).startswith("Sum of weights for aipw_0")]
    assert control_sums == [warned]
    assert np.isnan(actual_att)


@pytest.mark.filterwarnings("error")
def test_aipw_rc_basic_treated_unit_with_propensity_score_one(forbid_errstate):
    args = [arr.copy() for arr in ALL_VALID_ARGS_RC_BASIC]
    args[3][4] = 0.5
    expected = aipw_did_rc_imp1(*args)
    args[3][4] = 1.0
    forbid_errstate()

    assert aipw_did_rc_imp1(*args) == expected


def test_aipw_rc_basic_all_zero_i_weights():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[-1] = np.zeros_like(I_WEIGHTS_RC_UNIT_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Mean of i_weights is zero" in str(warn.message) for warn in w)
        term_names = ["aipw_1_pre", "aipw_1_post", "aipw_0_pre", "aipw_0_post"]
        for term_name in term_names:
            assert any(f"Sum of weights for {term_name} is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_non_finite_mean_i_weights():
    args_inf = list(ALL_VALID_ARGS_RC_BASIC)
    i_weights_inf = I_WEIGHTS_RC_UNIT_VALID.copy()
    i_weights_inf[0] = np.inf
    args_inf[-1] = i_weights_inf

    args_nan = list(ALL_VALID_ARGS_RC_BASIC)
    i_weights_nan = I_WEIGHTS_RC_UNIT_VALID.copy()
    i_weights_nan[0] = np.nan
    args_nan[-1] = i_weights_nan

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_inf = aipw_did_rc_imp1(*args_inf)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_inf, np.nan)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_nan = aipw_did_rc_imp1(*args_nan)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
    assert_allclose_with_nans(att_nan, np.nan)


@pytest.mark.parametrize("invalid_arg_idx", range(len(ALL_VALID_ARGS_RC_BASIC)))
def test_aipw_rc_basic_input_type_errors(invalid_arg_idx):
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[invalid_arg_idx] = list(args[invalid_arg_idx])
    with pytest.raises(TypeError, match="All inputs must be NumPy arrays."):
        aipw_did_rc_imp1(*args)


@pytest.mark.parametrize("mismatch_arg_idx", range(len(ALL_VALID_ARGS_RC_BASIC)))
def test_aipw_rc_basic_input_shape_mismatch_error(mismatch_arg_idx):
    args = list(ALL_VALID_ARGS_RC_BASIC)
    original_shape_len = args[mismatch_arg_idx].shape[0]
    if original_shape_len > 1:
        short_arr = np.array([1.0] * (original_shape_len - 1))
        args[mismatch_arg_idx] = short_arr
        with pytest.raises(ValueError, match="All input arrays must have the same shape."):
            aipw_did_rc_imp1(*args)
    elif original_shape_len == 1:
        long_arr = np.array([1.0] * (original_shape_len + 1))
        args[mismatch_arg_idx] = long_arr
        with pytest.raises(ValueError, match="All input arrays must have the same shape."):
            aipw_did_rc_imp1(*args)


@pytest.mark.parametrize("ndim_arg_idx", range(len(ALL_VALID_ARGS_RC_BASIC)))
def test_aipw_rc_basic_input_ndim_error(ndim_arg_idx):
    args = list(ALL_VALID_ARGS_RC_BASIC)
    arr_2d = np.array([args[ndim_arg_idx]])
    if args[ndim_arg_idx].size > 0:
        args[ndim_arg_idx] = arr_2d
        with pytest.raises(ValueError, match="All input arrays must be 1-dimensional."):
            aipw_did_rc_imp1(*args)


def test_aipw_rc_basic_empty_inputs():
    empty_arr = np.array([])
    empty_args = [empty_arr] * len(ALL_VALID_ARGS_RC_BASIC)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*empty_args)
        assert any("Mean of i_weights is not finite" in str(warn.message) for warn in w)
        term_names = ["aipw_1_pre", "aipw_1_post", "aipw_0_pre", "aipw_0_post"]
        for term_name in term_names:
            assert any(f"Sum of weights for {term_name} is 0.0" in str(warn.message) for warn in w) or any(
                f"Sum of weights for {term_name} is nan" in str(warn.message) for warn in w
            )
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_no_treated_units():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[2] = np.zeros_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Sum of weights for aipw_1_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for aipw_1_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_no_control_units():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[2] = np.ones_like(D_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Sum of weights for aipw_0_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for aipw_0_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_no_post_period():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[1] = np.zeros_like(POST_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Sum of weights for aipw_1_post is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for aipw_0_post is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_aipw_rc_basic_no_pre_period():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    args[1] = np.ones_like(POST_RC_VALID)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        actual_att = aipw_did_rc_imp1(*args)
        assert any("Sum of weights for aipw_1_pre is 0.0" in str(warn.message) for warn in w)
        assert any("Sum of weights for aipw_0_pre is 0.0" in str(warn.message) for warn in w)
    assert_allclose_with_nans(actual_att, np.nan)


def test_panel_trimming_effect():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0, 3.0, 2.0])
    d = np.array([1, 1, 0, 0, 1, 0])
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.65, 0.35])
    out_reg = np.array([0.5, 0.6, 0.2, 0.3, 0.55, 0.25])
    i_weights = np.ones_like(delta_y)

    att_no_trim = aipw_did_panel(delta_y, d, ps, out_reg, i_weights, trim_ps=None)

    trim_ps = np.array([1, 1, 1, 1, 0, 1])
    att_trimmed = aipw_did_panel(delta_y, d, ps, out_reg, i_weights, trim_ps)

    assert att_no_trim != att_trimmed


def test_panel_full_trimming():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0, 3.0, 2.0])
    d = np.array([1, 1, 0, 0, 1, 0])
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.65, 0.35])
    out_reg = np.array([0.5, 0.6, 0.2, 0.3, 0.55, 0.25])
    i_weights = np.ones_like(delta_y)
    trim_ps = np.zeros_like(delta_y)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_full_trim = aipw_did_panel(delta_y, d, ps, out_reg, i_weights, trim_ps)
        assert any("Sum of w_treat is zero" in str(warn.message) for warn in w)
    assert np.isnan(att_full_trim)


def test_panel_no_trimming_equivalence():
    delta_y = np.array([1.0, 2.0, 0.0, 1.0, 3.0, 2.0])
    d = np.array([1, 1, 0, 0, 1, 0])
    ps = np.array([0.6, 0.7, 0.3, 0.4, 0.65, 0.35])
    out_reg = np.array([0.5, 0.6, 0.2, 0.3, 0.55, 0.25])
    i_weights = np.ones_like(delta_y)

    att_none = aipw_did_panel(delta_y, d, ps, out_reg, i_weights, trim_ps=None)
    att_ones = aipw_did_panel(delta_y, d, ps, out_reg, i_weights, trim_ps=np.ones_like(delta_y))

    assert att_none == att_ones


def test_rc_basic_trimming_effect():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    att_no_trim = aipw_did_rc_imp1(*args, trim_ps=None)

    trim_ps = np.ones_like(args[0])
    trim_ps[0] = 0
    att_trimmed = aipw_did_rc_imp1(*args, trim_ps=trim_ps)

    assert att_no_trim != att_trimmed


def test_rc_basic_full_trimming():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    trim_ps = np.zeros_like(args[0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_full_trim = aipw_did_rc_imp1(*args, trim_ps=trim_ps)
        assert any("Sum of weights for aipw_1_pre is 0.0" in str(warn.message) for warn in w)
    assert np.isnan(att_full_trim)


def test_rc_basic_no_trimming_equivalence():
    args = list(ALL_VALID_ARGS_RC_BASIC)
    att_none = aipw_did_rc_imp1(*args, trim_ps=None)
    att_ones = aipw_did_rc_imp1(*args, trim_ps=np.ones_like(args[0]))

    assert att_none == att_ones


def test_rc_trimming_effect():
    args = list(ALL_VALID_ARGS_RC)
    att_no_trim = aipw_did_rc_imp2(*args, trim_ps=None)

    trim_ps = np.ones_like(args[0])
    trim_ps[0] = 0
    att_trimmed = aipw_did_rc_imp2(*args, trim_ps=trim_ps)

    assert att_no_trim != att_trimmed


def test_rc_full_trimming():
    args = list(ALL_VALID_ARGS_RC)
    trim_ps = np.zeros_like(args[0])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        att_full_trim = aipw_did_rc_imp2(*args, trim_ps=trim_ps)
        assert any("Sum of weights for att_treat_pre is 0.0" in str(warn.message) for warn in w)
    assert np.isnan(att_full_trim)


def test_rc_no_trimming_equivalence():
    args = list(ALL_VALID_ARGS_RC)
    att_none = aipw_did_rc_imp2(*args, trim_ps=None)
    att_ones = aipw_did_rc_imp2(*args, trim_ps=np.ones_like(args[0]))

    assert att_none == att_ones
