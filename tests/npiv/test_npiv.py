"""Tests for nonparametric instrumental variables estimation."""

import numpy as np
import polars as pl
import pytest

from moderndid.npiv.container import NPIVResult
from moderndid.npiv.npiv import npiv


def test_basic_npiv(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert isinstance(result, NPIVResult)
    assert result.h is not None
    assert len(result.h) == len(y)
    assert result.h_lower is not None
    assert result.h_upper is not None


def test_npiv_with_evaluation_points(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0, 1, 50).reshape(-1, 1)

    result = npiv(
        y=y,
        x=x,
        w=w,
        x_eval=x_eval,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert len(result.h) == len(x_eval)


def test_npiv_with_x_grid_compatibility(simple_data):
    y, x, w = simple_data
    x_grid = np.linspace(0, 1, 50).reshape(-1, 1)

    with pytest.warns(UserWarning, match="Using x_grid as x_eval"):
        result = npiv(
            y=y,
            x=x,
            w=w,
            x_grid=x_grid,
            j_x_segments=3,
            k_w_segments=4,
        )

    assert len(result.h) == len(x_grid)


@pytest.mark.parametrize("basis", ["tensor", "additive", "glp"])
def test_different_basis_types(simple_data, basis):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        basis=basis,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.h is not None


def test_derivative_estimation(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        ucb_deriv=True,
        deriv_index=1,
        deriv_order=1,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.deriv is not None
    assert result.h_lower_deriv is not None
    assert result.h_upper_deriv is not None


def test_multivariate_case(multivariate_data):
    y, x, w = multivariate_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.h is not None


def test_no_confidence_bands(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        ucb_h=False,
        ucb_deriv=False,
        j_x_segments=3,
        k_w_segments=4,
    )

    assert result.h is not None
    assert result.h_lower is None
    assert result.h_upper is None


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
def test_different_confidence_levels(simple_data, alpha):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        alpha=alpha,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.cv > 0


@pytest.mark.filterwarnings("ignore:Some 'x' values beyond boundary knots:UserWarning")
def test_with_range_constraints(simple_data):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        x_min=0.1,
        x_max=0.9,
        w_min=0.1,
        w_max=0.9,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.h is not None


def test_reproducibility_with_seed(simple_data):
    y, x, w = simple_data

    result1 = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
        seed=123,
    )

    result2 = npiv(
        y=y,
        x=x,
        w=w,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
        seed=123,
    )

    assert np.allclose(result1.h_lower, result2.h_lower)
    assert np.allclose(result1.h_upper, result2.h_upper)


@pytest.mark.parametrize("knots", ["uniform", "quantiles"])
def test_different_knot_types(simple_data, knots):
    y, x, w = simple_data

    result = npiv(
        y=y,
        x=x,
        w=w,
        knots=knots,
        j_x_segments=3,
        k_w_segments=4,
        biters=30,
    )

    assert result.h is not None


def test_input_validation():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 2))
    w = np.random.normal(0, 1, (n - 10, 2))

    with pytest.raises(ValueError, match="same number of observations"):
        npiv(y=y, x=x, w=w)


def test_invalid_alpha():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        npiv(y=y, x=x, w=w, alpha=1.5)


def test_invalid_basis():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="basis must be one of"):
        npiv(y=y, x=x, w=w, basis="invalid")


def test_invalid_deriv_index(multivariate_data):
    y, x, w = multivariate_data

    with pytest.raises(ValueError, match="deriv_index must be between"):
        npiv(y=y, x=x, w=w, deriv_index=3)


def test_small_sample_warning():
    n = 30
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.warns(UserWarning, match="Small sample size"):
        npiv(y=y, x=x, w=w, j_x_segments=2, k_w_segments=3)


def test_invalid_biters():
    n = 100
    y = np.random.normal(0, 1, n)
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    with pytest.raises(ValueError, match="biters must be positive"):
        npiv(y=y, x=x, w=w, biters=0)


def test_multidimensional_y():
    n = 100
    y = np.random.normal(0, 1, (n, 1))
    x = np.random.normal(0, 1, (n, 1))
    w = np.random.normal(0, 1, (n, 1))

    result = npiv(y=y, x=x, w=w, j_x_segments=3, k_w_segments=4)

    assert result.h is not None


def test_npiv_default_k_w_segments_refine_j(simple_data):
    y, x, w = simple_data

    default = npiv(y=y, x=x, w=w, j_x_segments=3, biters=30, seed=1)
    explicit = npiv(y=y, x=x, w=w, j_x_segments=3, k_w_segments=12, biters=30, seed=1)

    assert default.k_w_segments == 12
    np.testing.assert_array_equal(default.h, explicit.h)
    assert default.cv == explicit.cv


def test_npiv_instrument_basis_below_x_basis_raises(simple_data):
    y, x, w = simple_data

    with pytest.raises(ValueError, match="not identified"):
        npiv(y=y, x=x, w=w, j_x_segments=5, k_w_segments=2)


def test_npiv_negative_k_w_smooth_raises(simple_data):
    y, x, w = simple_data

    with pytest.raises(ValueError, match="k_w_smooth must be non-negative"):
        npiv(y=y, x=x, w=w, j_x_segments=3, k_w_smooth=-1)


def test_npiv_one_dimensional_inputs(simple_data):
    y, x, w = simple_data
    x_eval = np.linspace(0.1, 0.9, 25)

    flat = npiv(y=y, x=x.ravel(), w=w.ravel(), x_eval=x_eval, j_x_segments=3, biters=30, seed=4)
    column = npiv(y=y, x=x, w=w, x_eval=x_eval.reshape(-1, 1), j_x_segments=3, biters=30, seed=4)

    assert len(flat.h) == 25
    np.testing.assert_allclose(flat.h, column.h, rtol=1e-12)
    np.testing.assert_allclose(flat.h_upper, column.h_upper, rtol=1e-12)


def test_npiv_one_dimensional_eval_point_with_two_regressors(multivariate_data):
    y, x, w = multivariate_data

    result = npiv(y=y, x=x, w=w, x_eval=np.array([0.5, 0.5]), j_x_segments=2, ucb_h=False, ucb_deriv=False)

    assert result.h.shape == (1,)


def test_npiv_regression_selection_uses_x_basis(regression_data):
    y, x, w = regression_data

    result = npiv(y=y, x=x, w=w, biters=30, seed=2)

    assert result.k_w_degree == result.j_x_degree
    assert result.k_w_segments == result.j_x_segments
    np.testing.assert_array_equal(result.args["k_w_segments_set"], result.args["j_x_segments_set"])


def test_npiv_result_args_hold_no_internal_matrices(simple_data):
    y, x, w = simple_data

    result = npiv(y=y, x=x, w=w, biters=30, seed=3)

    assert not {"tmp", "psi_x_eval", "psi_x_deriv_eval", "b_w", "b_w_deriv"} & set(result.args)
    assert result.args["data_driven"] is True


# --- DataFrame API tests ---


def test_npiv_dataframe_polars():
    np.random.seed(42)
    n = 100
    df = pl.DataFrame(
        {
            "y": np.random.randn(n),
            "x": np.random.randn(n),
            "w": np.random.randn(n),
        }
    )
    result = npiv(data=df, yname="y", xname="x", wname="w", j_x_segments=3)
    assert isinstance(result, NPIVResult)
    assert len(result.h) == n


def test_npiv_dataframe_pandas():
    pd = pytest.importorskip("pandas")
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "y": np.random.randn(n),
            "x": np.random.randn(n),
            "w": np.random.randn(n),
        }
    )
    result = npiv(data=df, yname="y", xname="x", wname="w", j_x_segments=3)
    assert isinstance(result, NPIVResult)
    assert len(result.h) == n


def test_npiv_dataframe_error_both_data_and_arrays():
    n = 50
    df = pl.DataFrame({"y": np.random.randn(n), "x": np.random.randn(n), "w": np.random.randn(n)})
    with pytest.raises(ValueError, match="Cannot specify both"):
        npiv(data=df, yname="y", xname="x", wname="w", y=np.random.randn(n))


def test_npiv_dataframe_error_missing_column_names():
    n = 50
    df = pl.DataFrame({"y": np.random.randn(n), "x": np.random.randn(n), "w": np.random.randn(n)})
    with pytest.raises(ValueError, match="'yname', 'xname', and 'wname' are required"):
        npiv(data=df, yname="y", xname="x")


def test_npiv_dataframe_multivariate_columns():
    np.random.seed(42)
    n = 150
    df = pl.DataFrame(
        {
            "y": np.random.randn(n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "w1": np.random.randn(n),
            "w2": np.random.randn(n),
        }
    )
    result = npiv(data=df, yname="y", xname=["x1", "x2"], wname=["w1", "w2"], j_x_segments=3, biters=30)
    assert isinstance(result, NPIVResult)
    assert len(result.h) == n


def test_npiv_dataframe_no_data_no_arrays():
    with pytest.raises(ValueError, match="Must provide either"):
        npiv()
