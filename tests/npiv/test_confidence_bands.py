"""Tests for uniform confidence band construction."""

import numpy as np
import pytest

from moderndid.npiv.cck_ucb import compute_cck_ucb
from moderndid.npiv.confidence_bands import compute_ucb
from moderndid.npiv.container import NPIVResult
from moderndid.npiv.npiv import npiv
from moderndid.npiv.selection import npiv_choose_j
from moderndid.npiv.utils import _quantile_basis


def test_basic_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        alpha=0.05,
        biters=50,
    )

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert result.h_lower is not None
    assert result.h_upper is not None
    assert np.all(result.h_lower <= result.h)
    assert np.all(result.h <= result.h_upper)
    assert result.cv > 0


def test_confidence_bands_with_evaluation_points(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        biters=50,
    )

    assert len(result.h) == len(x_eval)
    assert len(result.h_lower) == len(x_eval)
    assert len(result.h_upper) == len(x_eval)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        basis=basis,
        biters=30,
    )

    assert result.h is not None
    assert result.h_lower is not None
    assert result.h_upper is not None


def test_derivative_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_index=1,
        deriv_order=1,
        biters=50,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None
    assert np.all(result.h_lower_deriv <= result.deriv)
    assert np.all(result.deriv <= result.h_upper_deriv)
    assert result.cv_deriv > 0


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        biters=30,
    )

    assert result.h is not None
    assert len(result.h) == len(y)


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_confidence_levels(simple_data, alpha):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        alpha=alpha,
        biters=50,
        seed=123,
    )

    coverage = np.mean(result.h_upper - result.h_lower)
    assert coverage > 0
    assert result.cv > 0


def test_no_confidence_bands(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=False,
    )

    assert result.h is not None
    assert result.h_lower is None
    assert result.h_upper is None
    assert result.h_lower_deriv is None
    assert result.h_upper_deriv is None
    assert result.cv is None
    assert result.cv_deriv is None


def test_function_bands_only(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=True,
        ucb_deriv=False,
        biters=50,
    )

    assert result.h_lower is not None
    assert result.h_upper is not None
    assert result.h_lower_deriv is None
    assert result.h_upper_deriv is None


def test_derivative_bands_only(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=True,
        biters=50,
    )

    assert result.h_lower is None
    assert result.h_upper is None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None


def test_with_fixed_dimensions(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        biters=50,
    )

    assert result.j_x_segments == 3
    assert result.k_w_segments == 4


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        biters=30,
    )

    assert result.h is not None


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = compute_ucb(
        y=y,
        x=x,
        w=w,
        biters=30,
        seed=123,
    )

    result2 = compute_ucb(
        y=y,
        x=x,
        w=w,
        biters=30,
        seed=123,
    )

    assert np.allclose(result1.h_lower, result2.h_lower)
    assert np.allclose(result1.h_upper, result2.h_upper)


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        knots=knots,
        biters=30,
    )

    assert result.h is not None


def test_with_selection_result(simple_data):
    y, x, w = simple_data

    selection_result = {
        "j_x_seg": 3,
        "k_w_seg": 4,
        "j_tilde": 5,
        "theta_star": 1.2,
        "j_x_segments_set": np.array([2, 3, 4, 5]),
        "k_w_segments_set": np.array([3, 4, 5, 6]),
    }

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        biters=50,
        selection_result=selection_result,
    )

    assert result.h is not None
    assert result.args["cck_method"] is True


def test_bootstrap_statistics_tracking(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        biters=100,
    )

    assert "boot_success_rate_h" in result.args
    assert "boot_success_rate_deriv" in result.args
    assert result.args["boot_success_rate_h"] >= 0.9


@pytest.mark.parametrize("j_x_degree,k_w_degree", [(2, 3), (3, 4), (4, 5)])
def test_different_spline_degrees(simple_data, j_x_degree, k_w_degree):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        j_x_degree=j_x_degree,
        k_w_degree=k_w_degree,
        biters=30,
    )

    assert result.j_x_degree == j_x_degree
    assert result.k_w_degree == k_w_degree


def test_higher_order_derivatives(simple_data):
    y, x, w = simple_data

    result = compute_ucb(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_order=2,
        biters=30,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None


@pytest.mark.parametrize(
    "q,expected",
    [
        (0.95, 9.22),
        (0.5, 3.1),
        (0.6056, 4.91312),
        (0.2, 0.97),
    ],
)
def test_bootstrap_quantile_interpolates_between_half_steps(q, expected):
    draws = np.array([3.1, 0.2, 5.5, 2.2, 4.8, 1.0, 9.3, 7.7, 6.1, 0.9, 2.5])

    assert _quantile_basis(draws, q) == pytest.approx(expected, rel=1e-12)


def test_default_k_w_segments_refine_j(simple_data):
    y, x, w = simple_data

    result = compute_ucb(y=y, x=x, w=w, j_x_segments=3, biters=30)
    smoother = compute_ucb(y=y, x=x, w=w, j_x_segments=3, k_w_smooth=0, biters=30)

    assert result.k_w_segments == 12
    assert smoother.k_w_segments == 3


def test_compute_ucb_routes_selection_without_segments(simple_data, five_point_selection):
    y, x, w = simple_data
    selection = five_point_selection(2)

    routed = compute_ucb(y=y, x=x, w=w, biters=20, seed=5, selection_result=selection)
    direct = compute_cck_ucb(y=y, x=x, w=w, biters=20, seed=5, selection_result=selection)

    assert routed.args["cck_method"] is True
    assert routed.j_x_segments == 2
    assert routed.cv == pytest.approx(direct.cv, rel=1e-12)
    np.testing.assert_allclose(routed.h_upper, direct.h_upper, rtol=1e-12)


@pytest.mark.parametrize("j_seg,expected", [(2, 3), (8, 4)])
def test_cck_bootstrap_set_holds_selected_dimension(simple_data, five_point_selection, j_seg, expected):
    y, x, w = simple_data

    result = compute_cck_ucb(y=y, x=x, w=w, biters=30, seed=3, selection_result=five_point_selection(j_seg))

    assert result.args["n_j_boot"] == expected


@pytest.mark.parametrize("j_seg", [2, 8])
def test_cck_critical_value_has_no_envelope_branch(simple_data, five_point_selection, j_seg):
    y, x, w = simple_data
    margin = np.log(np.log(j_seg + 3)) * 1.3

    full = compute_cck_ucb(y=y, x=x, w=w, biters=30, seed=3, selection_result=five_point_selection(j_seg))
    bare = compute_cck_ucb(
        y=y, x=x, w=w, biters=30, seed=3, selection_result=five_point_selection(j_seg, theta_star=0.0)
    )

    assert full.cv == pytest.approx(bare.cv + margin, rel=1e-12)
    assert full.cv_deriv == pytest.approx(bare.cv_deriv + margin, rel=1e-12)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("j_tilde", [1, 2])
def test_cck_dimension_below_e_adds_no_margin(simple_data, five_point_selection, j_tilde):
    y, x, w = simple_data
    selection = {**five_point_selection(1), "j_tilde": j_tilde}

    full = compute_cck_ucb(y=y, x=x, w=w, biters=30, seed=3, selection_result=selection)
    bare = compute_cck_ucb(y=y, x=x, w=w, biters=30, seed=3, selection_result={**selection, "theta_star": 0.0})

    assert full.cv == bare.cv
    assert full.cv_deriv == bare.cv_deriv


def test_cck_regression_ignores_instrument_arguments(regression_data):
    y, x, w = regression_data
    selection = npiv_choose_j(y=y, x=x, w=w, biters=30, seed=4)

    passed = compute_cck_ucb(y=y, x=x, w=w, k_w_degree=4, biters=30, seed=4, selection_result=selection)
    aligned = compute_cck_ucb(y=y, x=x, w=w, k_w_degree=3, biters=30, seed=4, selection_result=selection)
    via_npiv = npiv(y=y, x=x, w=w, biters=30, seed=4)

    assert passed.cv == pytest.approx(aligned.cv, rel=1e-12)
    assert passed.cv == pytest.approx(via_npiv.cv, rel=1e-12)
    assert passed.cv_deriv == pytest.approx(via_npiv.cv_deriv, rel=1e-12)
    np.testing.assert_allclose(passed.h_upper, via_npiv.h_upper, rtol=1e-12)


def test_cck_without_selection_result_raises(simple_data):
    y, x, w = simple_data

    with pytest.raises(ValueError, match="needs selection_result"):
        compute_cck_ucb(y=y, x=x, w=w, biters=10)
