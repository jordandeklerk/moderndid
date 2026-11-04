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


@pytest.mark.parametrize(
    "name",
    ["y", "post", "d", "ps", "out_y_treat_post", "out_y_treat_pre", "out_y_cont_post", "out_y_cont_pre", "i_weights"],
)
def test_aipw_rc_input_ndim_error(aipw_rc_imp2_arrays, name):
    aipw_rc_imp2_arrays[name] = aipw_rc_imp2_arrays[name][None, :]

    with pytest.raises(ValueError, match=re.escape("All input arrays must be 1-dimensional.")):
        aipw_did_rc_imp2(**aipw_rc_imp2_arrays)


def test_aipw_rc_empty_inputs():
    terms = ["att_treat_pre", "att_treat_post", "att_cont_pre", "att_cont_post"]
    terms += ["att_d_post", "att_dt1_post", "att_d_pre", "att_dt0_pre"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(*[np.array([])] * 9)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is not finite. Using original weights.",
        *[f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms],
    ]
    assert np.isnan(att)


def test_aipw_rc_specific_term_nan_due_to_zero_weights(aipw_rc_imp2_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**{**aipw_rc_imp2_arrays, "post": np.ones(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for att_treat_pre is 0.0. Term will be NaN.",
        "Sum of weights for att_cont_pre is 0.0. Term will be NaN.",
        "Sum of weights for att_dt0_pre is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_no_treated_units(aipw_rc_imp2_arrays):
    terms = ["att_treat_pre", "att_treat_post", "att_d_post", "att_dt1_post", "att_d_pre", "att_dt0_pre"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**{**aipw_rc_imp2_arrays, "d": np.zeros(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms
    ]
    assert np.isnan(att)


def test_aipw_rc_no_control_units(aipw_rc_imp2_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**{**aipw_rc_imp2_arrays, "d": np.ones(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for att_cont_pre is 0.0. Term will be NaN.",
        "Sum of weights for att_cont_post is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_basic_happy_path_unit_weights(aipw_rc_imp1_arrays):
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)

    att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    assert isinstance(att, float)
    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_aipw_rc_basic_happy_path_non_uniform_weights(aipw_rc_imp1_arrays):
    i_weights = np.array([2.0, 1.0, 3.0, 1.0, 1.0, 2.0, 1.0, 3.0, 3.0, 2.0])
    treated = (1 * 7 + 3 * 4 + 2 * 10) / 6 - (1 * 3 + 2 * 5 + 3 * 1) / 6
    control = (3 * (1 / 3) * 1 + 1 * 4 * 6) / (3 * (1 / 3) + 1 * 4) - (2 * 1 * 2 + 1 * 3 * -2) / (2 * 1 + 1 * 3)

    att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "i_weights": i_weights})

    assert isinstance(att, float)
    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_aipw_rc_basic_ps_one_for_control_unit(aipw_rc_imp1_arrays):
    aipw_rc_imp1_arrays["ps"][[0, 2]] = 1.0

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Propensity score is 1 for some control units. Their weights will be NaN/Inf. "
        "This typically indicates issues with the propensity score model.",
        "Sum of weights for aipw_0_pre is inf. Term will be NaN.",
        "Sum of weights for aipw_0_post is inf. Term will be NaN.",
    ]
    assert np.isnan(att)


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
    aipw_rc_imp1_arrays, forbid_errstate, unit, weight, warned
):
    aipw_rc_imp1_arrays["ps"][unit] = 1.0
    aipw_rc_imp1_arrays["i_weights"][unit] = weight
    forbid_errstate()

    with pytest.warns(UserWarning) as record:
        actual_att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    control_sums = [str(warn.message) for warn in record if str(warn.message).startswith("Sum of weights for aipw_0")]
    assert control_sums == [warned]
    assert np.isnan(actual_att)


@pytest.mark.filterwarnings("error")
def test_aipw_rc_basic_treated_unit_with_propensity_score_one(aipw_rc_imp1_arrays, forbid_errstate):
    aipw_rc_imp1_arrays["ps"][4] = 1.0
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)
    forbid_errstate()

    att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_aipw_rc_basic_all_zero_i_weights(aipw_rc_imp1_arrays):
    terms = ["aipw_1_pre", "aipw_1_post", "aipw_0_pre", "aipw_0_post"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "i_weights": np.zeros(10)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is zero, cannot normalize. Using original weights.",
        *[f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms],
    ]
    assert np.isnan(att)


@pytest.mark.parametrize("weight", [np.inf, np.nan])
def test_aipw_rc_basic_non_finite_mean_i_weights(aipw_rc_imp1_arrays, weight):
    aipw_rc_imp1_arrays["i_weights"][0] = weight

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    assert "Mean of i_weights is not finite. Using original weights." in [str(w.message) for w in record]
    assert np.isnan(att)


@pytest.mark.parametrize("name", ["y", "post", "d", "ps", "out_reg", "i_weights"])
def test_aipw_rc_basic_input_type_errors(aipw_rc_imp1_arrays, name):
    aipw_rc_imp1_arrays[name] = aipw_rc_imp1_arrays[name].tolist()

    with pytest.raises(TypeError, match=re.escape("All inputs must be NumPy arrays.")):
        aipw_did_rc_imp1(**aipw_rc_imp1_arrays)


@pytest.mark.parametrize("name", ["y", "post", "d", "ps", "out_reg", "i_weights"])
def test_aipw_rc_basic_input_shape_mismatch_error(aipw_rc_imp1_arrays, name):
    aipw_rc_imp1_arrays[name] = aipw_rc_imp1_arrays[name][:-1]

    with pytest.raises(ValueError, match=re.escape("All input arrays must have the same shape.")):
        aipw_did_rc_imp1(**aipw_rc_imp1_arrays)


@pytest.mark.parametrize("name", ["y", "post", "d", "ps", "out_reg", "i_weights"])
def test_aipw_rc_basic_input_ndim_error(aipw_rc_imp1_arrays, name):
    aipw_rc_imp1_arrays[name] = aipw_rc_imp1_arrays[name][None, :]

    with pytest.raises(ValueError, match=re.escape("All input arrays must be 1-dimensional.")):
        aipw_did_rc_imp1(**aipw_rc_imp1_arrays)


def test_aipw_rc_basic_empty_inputs():
    terms = ["aipw_1_pre", "aipw_1_post", "aipw_0_pre", "aipw_0_post"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(*[np.array([])] * 6)

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Mean of i_weights is not finite. Using original weights.",
        *[f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms],
    ]
    assert np.isnan(att)


def test_aipw_rc_basic_missing_outcome_gives_nan_with_a_warning(aipw_rc_imp1_arrays):
    aipw_rc_imp1_arrays["y"][6] = np.nan

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays)

    assert "Weighted sum for aipw_1_post is not finite (nan). Term will be NaN." in [str(w.message) for w in record]
    assert np.isnan(att)


def test_aipw_rc_basic_no_treated_units(aipw_rc_imp1_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "d": np.zeros(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for aipw_1_pre is 0.0. Term will be NaN.",
        "Sum of weights for aipw_1_post is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_basic_no_control_units(aipw_rc_imp1_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "d": np.ones(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for aipw_0_pre is 0.0. Term will be NaN.",
        "Sum of weights for aipw_0_post is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_basic_no_post_period(aipw_rc_imp1_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "post": np.zeros(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for aipw_1_post is 0.0. Term will be NaN.",
        "Sum of weights for aipw_0_post is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_aipw_rc_basic_no_pre_period(aipw_rc_imp1_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**{**aipw_rc_imp1_arrays, "post": np.ones(10, dtype=int)})

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of weights for aipw_1_pre is 0.0. Term will be NaN.",
        "Sum of weights for aipw_0_pre is 0.0. Term will be NaN.",
    ]
    assert np.isnan(att)


def test_panel_trimming_effect(aipw_panel_arrays):
    trim_ps = np.array([1, 1, 1, 1, 0, 0])
    treated = (0.5 + 1.4) / 2
    control = (3 / 7 * -0.2 + 2 / 3 * 0.7) / (3 / 7 + 2 / 3)

    att = aipw_did_panel(**aipw_panel_arrays, trim_ps=trim_ps)

    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_panel_full_trimming(aipw_panel_arrays):
    with pytest.warns(UserWarning) as record:
        att = aipw_did_panel(**aipw_panel_arrays, trim_ps=np.zeros(6))

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        "Sum of w_treat is zero (no effectively treated units). aipw_1 will be NaN.",
        "Sum of w_cont is 0.0 (no effectively control units or problematic weights). aipw_0 will be NaN.",
    ]
    assert np.isnan(att)


def test_panel_no_trimming_equivalence(aipw_panel_arrays):
    treated = (0.5 + 1.4 + 2.45) / 3
    control = (3 / 7 * -0.2 + 2 / 3 * 0.7 + 7 / 13 * 1.75) / (3 / 7 + 2 / 3 + 7 / 13)

    att_none = aipw_did_panel(**aipw_panel_arrays, trim_ps=None)
    att_ones = aipw_did_panel(**aipw_panel_arrays, trim_ps=np.ones(6))

    assert att_none == att_ones
    np.testing.assert_allclose(att_ones, treated - control, rtol=1e-12, atol=0)


def test_rc_basic_trimming_effect(aipw_rc_imp1_arrays):
    trim_ps = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    treated = (7 + 4) / 2 - (3 + 5) / 2
    control = 1 - 2

    att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays, trim_ps=trim_ps)

    np.testing.assert_allclose(att, treated - control, rtol=1e-12, atol=0)


def test_rc_basic_full_trimming(aipw_rc_imp1_arrays):
    terms = ["aipw_1_pre", "aipw_1_post", "aipw_0_pre", "aipw_0_post"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp1(**aipw_rc_imp1_arrays, trim_ps=np.zeros(10))

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms
    ]
    assert np.isnan(att)


def test_rc_basic_no_trimming_equivalence(aipw_rc_imp1_arrays):
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)

    att_none = aipw_did_rc_imp1(**aipw_rc_imp1_arrays, trim_ps=None)
    att_ones = aipw_did_rc_imp1(**aipw_rc_imp1_arrays, trim_ps=np.ones(10))

    assert att_none == att_ones
    np.testing.assert_allclose(att_ones, treated - control, rtol=1e-12, atol=0)


def test_rc_trimming_effect(aipw_rc_imp2_arrays):
    trim_ps = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 0])
    treated = (7 + 4) / 2 - (3 + 5) / 2
    control = 1 - 2
    efficiency_post = (2 + 4 + 1 + 3) / 4 - (1 + 3) / 2
    efficiency_pre = (1 + 2 + 4 + 2) / 4 - (1 + 2) / 2

    att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays, trim_ps=trim_ps)

    np.testing.assert_allclose(att, treated - control + efficiency_post - efficiency_pre, rtol=1e-12, atol=0)


def test_rc_full_trimming(aipw_rc_imp2_arrays):
    terms = ["att_treat_pre", "att_treat_post", "att_cont_pre", "att_cont_post"]
    terms += ["att_d_post", "att_dt1_post", "att_d_pre", "att_dt0_pre"]

    with pytest.warns(UserWarning) as record:
        att = aipw_did_rc_imp2(**aipw_rc_imp2_arrays, trim_ps=np.zeros(10))

    assert [str(w.message) for w in record if w.category is UserWarning] == [
        f"Sum of weights for {term} is 0.0. Term will be NaN." for term in terms
    ]
    assert np.isnan(att)


def test_rc_no_trimming_equivalence(aipw_rc_imp2_arrays):
    treated = (7 + 4 + 10) / 3 - (3 + 5 + 1) / 3
    control = (1 / 3 * 1 + 4 * 6) / (1 / 3 + 4) - (1 * 2 + 3 * -2) / (1 + 3)
    efficiency_post = (2 + 4 + 1 + 3 + 6 + 5) / 6 - (1 + 3 + 5) / 3
    efficiency_pre = (1 + 2 + 4 + 2 + 3 + 5) / 6 - (1 + 2 + 3) / 3

    att_none = aipw_did_rc_imp2(**aipw_rc_imp2_arrays, trim_ps=None)
    att_ones = aipw_did_rc_imp2(**aipw_rc_imp2_arrays, trim_ps=np.ones(10))

    assert att_none == att_ones
    np.testing.assert_allclose(att_ones, treated - control + efficiency_post - efficiency_pre, rtol=1e-12, atol=0)
