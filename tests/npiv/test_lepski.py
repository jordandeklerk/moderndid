"""Tests for Lepski method for optimal dimension selection."""

import numpy as np
import pytest

from moderndid.npiv import lepski
from moderndid.npiv.lepski import npiv_j, npiv_jhat_max


def test_basic_lepski_selection(simple_data):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
        alpha=0.5,
    )

    assert isinstance(result, dict)
    assert "j_tilde" in result
    assert "j_hat" in result
    assert "j_hat_n" in result
    assert "j_x_seg" in result
    assert "k_w_seg" in result
    assert "theta_star" in result

    assert result["j_tilde"] > 0
    assert result["theta_star"] > 0


def test_lepski_with_custom_segments(simple_data):
    y, x, w = simple_data

    j_x_segments_set = np.array([2, 4, 8])
    k_w_segments_set = np.array([3, 5, 9])

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=j_x_segments_set,
        k_w_segments_set=k_w_segments_set,
        biters=30,
    )

    assert result["j_x_seg"] in j_x_segments_set
    assert result["k_w_seg"] in k_w_segments_set


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        basis=basis,
        biters=20,
    )

    assert result["j_tilde"] > 0


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
    )

    assert result["j_tilde"] > 0


def test_regression_case(regression_data):
    y, x, w = regression_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
    )

    assert result["j_tilde"] > 0


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_different_alpha_levels(simple_data, alpha):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        alpha=alpha,
        biters=20,
    )

    assert result["theta_star"] > 0


def test_with_custom_grid(simple_data):
    y, x, w = simple_data

    x_grid = np.linspace(0, 1, 30).reshape(-1, 1)

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        x_grid=x_grid,
        grid_num=30,
        biters=20,
    )

    assert result["j_tilde"] > 0


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        biters=20,
    )

    assert result["j_tilde"] > 0


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
        seed=123,
    )

    result2 = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
        seed=123,
    )

    assert result1["j_tilde"] == result2["j_tilde"]
    assert result1["theta_star"] == result2["theta_star"]


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        knots=knots,
        biters=20,
    )

    assert result["j_tilde"] > 0


def test_test_matrix_structure(simple_data):
    y, x, w = simple_data

    result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=np.array([2, 4]),
        k_w_segments_set=np.array([3, 5]),
        biters=20,
    )

    assert "test_matrix" in result
    assert "z_sup" in result
    test_mat = result["test_matrix"]
    assert test_mat.shape[0] == test_mat.shape[1]
    assert np.all(np.isnan(test_mat) | (test_mat >= 0))


def test_jhat_max_basic(simple_data):
    _, x, w = simple_data

    result = npiv_jhat_max(x=x, w=w)

    assert isinstance(result, dict)
    assert "j_x_segments_set" in result
    assert "k_w_segments_set" in result
    assert "j_hat_max" in result
    assert "alpha_hat" in result

    assert result["j_hat_max"] > 0
    assert 0 < result["alpha_hat"] <= 0.5
    assert len(result["j_x_segments_set"]) == len(result["k_w_segments_set"])


def test_jhat_max_regression_case(regression_data):
    _, x, w = regression_data

    result = npiv_jhat_max(x=x, w=w)

    assert result["j_hat_max"] > 0


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_jhat_max_basis_types(simple_data, basis):
    _, x, w = simple_data

    result = npiv_jhat_max(x=x, w=w, basis=basis)

    assert result["j_hat_max"] > 0


def test_jhat_max_multivariate(multivariate_data):
    _, x, w = multivariate_data

    result = npiv_jhat_max(x=x, w=w)

    assert result["j_hat_max"] > 0


def test_jhat_max_small_sample():
    n = 50
    x = np.random.uniform(0, 1, (n, 1))
    w = np.random.uniform(0, 1, (n, 1))

    result = npiv_jhat_max(x=x, w=w)

    assert result["j_hat_max"] > 0
    assert len(result["j_x_segments_set"]) >= 1


def test_integrated_lepski_workflow(simple_data):
    y, x, w = simple_data

    jhat_result = npiv_jhat_max(x=x, w=w)

    lepski_result = npiv_j(
        y=y,
        x=x,
        w=w,
        j_x_segments_set=jhat_result["j_x_segments_set"][:2],
        k_w_segments_set=jhat_result["k_w_segments_set"][:2],
        alpha=jhat_result["alpha_hat"],
        biters=20,
    )

    assert lepski_result["j_tilde"] > 0
    assert lepski_result["j_tilde"] <= jhat_result["j_hat_max"]


def test_jhat_max_stops_where_sieve_measure_is_zero(simple_data, monkeypatch):
    _, x, w = simple_data
    monkeypatch.setattr(lepski, "_compute_sieve_measure", lambda *args: 0.0 if args[2] >= 4 else 0.5)

    result = npiv_jhat_max(x=x, w=w)

    np.testing.assert_array_equal(result["j_x_segments_set"], [1, 2])
    assert result["j_hat_max"] == 5


def test_sieve_measure_failure_returns_zero(simple_data, monkeypatch):
    _, x, w = simple_data

    def broken_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(np.linalg, "svd", broken_svd)

    assert (
        lepski._compute_sieve_measure(x, w, 2, 8, 3, 4, 1, 1, "uniform", "tensor", None, None, None, None, 200) == 0.0
    )


def test_lepski_zero_support_bound_is_used(simple_data):
    y, x, w = simple_data
    kwargs = {"j_x_segments_set": np.array([2, 4]), "k_w_segments_set": np.array([8, 16]), "biters": 20, "seed": 1}

    zero = npiv_j(y=y, x=x, w=w, w_min=0.0, w_max=1.0, **kwargs)
    tiny = npiv_j(y=y, x=x, w=w, w_min=1e-300, w_max=1.0, **kwargs)
    data = npiv_j(y=y, x=x, w=w, w_max=1.0, **kwargs)

    assert zero["theta_star"] == pytest.approx(tiny["theta_star"], rel=1e-9)
    assert zero["theta_star"] != data["theta_star"]


def test_lepski_additive_matches_tensor_with_one_regressor(simple_data):
    y, x, w = simple_data
    kwargs = {
        "j_x_segments_set": np.array([1, 2, 4]),
        "k_w_segments_set": np.array([4, 8, 16]),
        "biters": 30,
        "seed": 2,
    }

    tensor = npiv_j(y=y, x=x, w=w, basis="tensor", **kwargs)
    additive = npiv_j(y=y, x=x, w=w, basis="additive", **kwargs)

    assert additive["theta_star"] == pytest.approx(tensor["theta_star"], rel=1e-8)
    np.testing.assert_allclose(additive["z_sup"], tensor["z_sup"], rtol=1e-8)
    assert additive["j_x_seg"] == tensor["j_x_seg"]


def test_lepski_sieve_measure_additive_matches_tensor_with_one_regressor(simple_data):
    _, x, w = simple_data
    common = (x, w, 2, 8, 3, 4, 1, 1, "uniform")

    tensor = lepski._compute_sieve_measure(*common, "tensor", None, None, None, None, len(x))
    additive = lepski._compute_sieve_measure(*common, "additive", None, None, None, None, len(x))

    assert additive == pytest.approx(tensor, rel=1e-8)


def test_lepski_unidentified_dimension_raises(simple_data):
    y, x, w = simple_data

    with pytest.raises(ValueError, match="not identified"):
        npiv_j(y=y, x=x, w=w, j_x_segments_set=np.array([2, 8]), k_w_segments_set=np.array([3, 4]), biters=10)
