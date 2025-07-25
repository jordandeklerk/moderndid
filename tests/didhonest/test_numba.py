# pylint: disable=redefined-outer-name,protected-access
"""Tests for Numba optimizations."""

import time

import numpy as np
import pytest

from didpy.didhonest import arp_no_nuisance, arp_nuisance, bounds, numba, utils


def _create_first_differences_matrix_py(num_pre_periods, num_post_periods):
    total_periods = num_pre_periods + num_post_periods + 1
    a_tilde = np.zeros((num_pre_periods + num_post_periods, total_periods))
    for r in range(num_pre_periods + num_post_periods):
        a_tilde[r, r : (r + 2)] = [-1, 1]
    return a_tilde


def _create_second_difference_matrix_py(num_pre_periods, num_post_periods):
    total_periods = num_pre_periods + num_post_periods
    n_pre_diffs = num_pre_periods - 2 if num_pre_periods >= 3 else 0
    n_post_diffs = num_post_periods - 2 if num_post_periods >= 3 else 0
    n_diffs = n_pre_diffs + n_post_diffs
    if n_diffs == 0:
        return np.zeros((0, total_periods))
    a_sd = np.zeros((n_diffs, total_periods))
    if n_pre_diffs > 0:
        for r in range(n_pre_diffs):
            a_sd[r, r : (r + 3)] = [1, -2, 1]
    if n_post_diffs > 0:
        for r in range(n_post_diffs):
            post_idx = n_pre_diffs + r
            coef_idx = num_pre_periods + r
            a_sd[post_idx, coef_idx : (coef_idx + 3)] = [1, -2, 1]
    return a_sd


def _create_monotonicity_matrix_py(num_pre_periods, num_post_periods):
    total_periods = num_pre_periods + num_post_periods
    a_m = np.zeros((total_periods, total_periods))
    for r in range(num_pre_periods - 1):
        a_m[r, r : (r + 2)] = [1, -1]
    a_m[num_pre_periods - 1, num_pre_periods - 1] = 1
    if num_post_periods > 0:
        a_m[num_pre_periods, num_pre_periods] = -1
        if num_post_periods > 1:
            for r in range(num_pre_periods + 1, num_pre_periods + num_post_periods):
                a_m[r, (r - 1) : r + 1] = [1, -1]
    return a_m


def _selection_matrix_py(selection_0idx, size, n_selections, select_rows):
    if select_rows:
        m = np.zeros((n_selections, size))
        for i, idx in enumerate(selection_0idx):
            m[i, idx] = 1
    else:
        m = np.zeros((size, n_selections))
        for i, idx in enumerate(selection_0idx):
            m[idx, i] = 1
    return m


def _compute_bounds_py(eta, sigma, A, b, z):
    c = numba.lee_coefficient(eta, sigma)
    Az = A @ z
    Ac = A @ c
    nonzero_mask = np.abs(Ac) > 1e-10
    objective = np.full_like(Ac, np.nan)
    objective[nonzero_mask] = (b[nonzero_mask] - Az[nonzero_mask]) / Ac[nonzero_mask]
    ac_negative_idx = Ac < 0
    ac_positive_idx = Ac > 0
    lower_bound = np.max(objective[ac_negative_idx]) if np.any(ac_negative_idx) else -np.inf
    upper_bound = np.min(objective[ac_positive_idx]) if np.any(ac_positive_idx) else np.inf
    return lower_bound, upper_bound


def _test_over_theta_grid_py(
    beta_hat, sigma, A, d, theta_grid, n_pre_periods, post_period_index, alpha, test_fn, **test_kwargs
):
    results = []
    post_period_vec = utils.basis_vector(index=n_pre_periods + post_period_index, size=len(beta_hat)).flatten()
    for theta in theta_grid:
        y = beta_hat - post_period_vec * theta
        in_set = test_fn(y=y, sigma=sigma, A=A, d=d, alpha=alpha, **test_kwargs)
        results.append([theta, float(in_set)])
    return np.array(results)


@pytest.fixture
def random_data():
    np.random.seed(42)
    n = 10
    eta = np.random.randn(n)
    sigma = np.eye(n) + 0.1 * np.random.randn(n, n)
    sigma = sigma @ sigma.T
    A = np.random.randn(5, n)
    b = np.random.randn(5)
    z = np.random.randn(n)
    return eta, sigma, A, b, z


@pytest.fixture
def grid_search_data():
    np.random.seed(42)
    n = 10
    beta_hat = np.random.randn(n)
    sigma = np.eye(n) + 0.1 * np.random.randn(n, n)
    sigma = sigma @ sigma.T
    A = np.random.randn(5, n)
    d = np.random.randn(5)
    theta_grid = np.linspace(-2, 2, 20)
    n_pre_periods = 5
    post_period_index = 1
    alpha = 0.05
    return beta_hat, sigma, A, d, theta_grid, n_pre_periods, post_period_index, alpha


@pytest.fixture(params=[(10, 10), (20, 20)])
def matrix_construction_sizes(request, fast_config):
    if fast_config["skip_expensive_params"] and request.param == (20, 20):
        pytest.skip("Skipping expensive parameter combination")
    return request.param


@pytest.fixture(params=[30, 50])
def grid_search_sizes(request, fast_config):
    if fast_config["skip_expensive_params"] and request.param == 50:
        pytest.skip("Skipping expensive parameter combination")
    np.random.seed(42)
    n = fast_config["n_medium"]
    grid_size = request.param
    data = {
        "beta_hat": np.random.randn(n),
        "sigma": np.eye(n) + 0.1 * np.random.randn(n, n),
        "A": np.random.randn(5, n),
        "d": np.random.randn(5),
        "theta_grid": np.linspace(-2, 2, grid_size),
        "n_pre_periods": min(n // 2, 10),
        "post_period_index": 1,
        "alpha": 0.05,
        "grid_size": grid_size,
    }
    data["sigma"] = data["sigma"] @ data["sigma"].T
    return data


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_compute_bounds_consistency(random_data):
    eta, sigma, A, b, z = random_data
    bounds_numba = numba.compute_bounds(eta, sigma, A, b, z)
    bounds_original = _compute_bounds_py(eta, sigma, A, b, z)
    np.testing.assert_allclose(bounds_numba[0], bounds_original[0], rtol=1e-10)
    np.testing.assert_allclose(bounds_numba[1], bounds_original[1], rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize("select", ["rows", "columns"])
def test_selection_matrix_consistency(select):
    selection = np.array([2, 4, 6])
    size = 10
    matrix_numba = numba.selection_matrix(selection, size, select)
    matrix_original = _selection_matrix_py(selection - 1, size, len(selection), select == "rows")
    np.testing.assert_array_equal(matrix_numba, matrix_original)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_first_differences_matrix_consistency():
    num_pre, num_post = 5, 4
    matrix_numba = numba.create_first_differences_matrix(num_pre, num_post)
    matrix_original = _create_first_differences_matrix_py(num_pre, num_post)
    np.testing.assert_array_equal(matrix_numba, matrix_original)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize("num_pre,num_post", [(2, 2), (3, 3), (5, 4)])
def test_second_difference_matrix_consistency(num_pre, num_post):
    matrix_numba = bounds.create_second_difference_matrix(num_pre, num_post)
    matrix_original = _create_second_difference_matrix_py(num_pre, num_post)
    np.testing.assert_array_equal(matrix_numba, matrix_original)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize("num_pre,num_post", [(2, 1), (3, 2), (5, 4)])
def test_monotonicity_matrix_consistency(num_pre, num_post):
    matrix_numba = bounds._create_monotonicity_matrix_impl(num_pre, num_post)
    matrix_original = _create_monotonicity_matrix_py(num_pre, num_post)
    np.testing.assert_array_equal(matrix_numba, matrix_original)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_grid_search_consistency(grid_search_data):
    results_numba = arp_no_nuisance._test_over_theta_grid(
        *grid_search_data, test_fn=arp_no_nuisance._test_in_identified_set
    )
    results_original = _test_over_theta_grid_py(*grid_search_data, test_fn=arp_no_nuisance._test_in_identified_set)
    np.testing.assert_array_equal(results_numba, results_original)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_y_values_preparation():
    np.random.seed(42)
    y, a_gamma_inv_one = np.random.randn(10), np.random.randn(10)
    theta_grid = np.linspace(-1, 1, 5)
    y_matrix_numba = arp_nuisance.prepare_theta_grid_y_values(y, a_gamma_inv_one, theta_grid)
    y_matrix_original = np.array([y - a_gamma_inv_one * theta for theta in theta_grid])
    np.testing.assert_allclose(y_matrix_numba, y_matrix_original, rtol=1e-10)


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
def test_hybrid_dbar_computation():
    flci_halflength, vbar, d_vec = 0.5, np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0])
    a_gamma_inv_one, theta = np.array([0.4, 0.5, 0.6]), 0.7
    dbar_numba = arp_nuisance.compute_hybrid_dbar(flci_halflength, vbar, d_vec, a_gamma_inv_one, theta)
    vbar_d, vbar_a = np.dot(vbar, d_vec), np.dot(vbar, a_gamma_inv_one)
    dbar_manual = np.array(
        [flci_halflength - vbar_d + (1 - vbar_a) * theta, flci_halflength + vbar_d - (1 - vbar_a) * theta]
    )
    np.testing.assert_allclose(dbar_numba, dbar_manual, rtol=1e-10)


def time_function(func, *args, **kwargs):
    start = time.perf_counter()
    func(*args, **kwargs)
    end = time.perf_counter()
    return end - start


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.perf
def test_compute_bounds_performance(random_data, request):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    eta, sigma, A, b, z = random_data
    numba.compute_bounds(eta, sigma, A, b, z)
    time_numba = time_function(numba.compute_bounds, eta, sigma, A, b, z)
    time_original = time_function(_compute_bounds_py, eta, sigma, A, b, z)
    assert time_original > time_numba


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize("select", ["rows", "columns"])
@pytest.mark.perf
def test_selection_matrix_performance(select, request, fast_config):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    selection_size = fast_config["n_medium"]
    selection, size = np.arange(1, selection_size + 1), selection_size * 2
    numba.selection_matrix(selection, size, select)
    time_numba = time_function(numba.selection_matrix, selection, size, select)
    time_original = time_function(_selection_matrix_py, selection - 1, size, len(selection), select == "rows")
    assert time_original > time_numba


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.perf
def test_second_difference_matrix_performance(matrix_construction_sizes, request):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    num_pre, num_post = matrix_construction_sizes
    bounds.create_second_difference_matrix(num_pre, num_post)  # Warm-up
    time_numba = time_function(bounds.create_second_difference_matrix, num_pre, num_post)
    time_original = time_function(_create_second_difference_matrix_py, num_pre, num_post)
    if num_pre > 10:
        assert time_original * 1.1 > time_numba, "Numba version should not be significantly slower"


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.perf
def test_monotonicity_matrix_performance(matrix_construction_sizes, request):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    num_pre, num_post = matrix_construction_sizes
    bounds._create_monotonicity_matrix_impl(num_pre, num_post)  # Warm-up
    time_numba = time_function(bounds._create_monotonicity_matrix_impl, num_pre, num_post)
    time_original = time_function(_create_monotonicity_matrix_py, num_pre, num_post)
    if num_pre > 10:
        assert time_original * 1.1 > time_numba, "Numba version should not be significantly slower"


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.perf
def test_first_differences_matrix_performance(matrix_construction_sizes, request):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    num_pre, num_post = matrix_construction_sizes
    numba.create_first_differences_matrix(num_pre, num_post)
    time_numba = time_function(numba.create_first_differences_matrix, num_pre, num_post)
    time_original = time_function(_create_first_differences_matrix_py, num_pre, num_post)
    if num_pre > 10:
        assert time_original > time_numba


@pytest.mark.skipif(not numba.HAS_NUMBA, reason="Numba not available")
@pytest.mark.perf
def test_grid_search_performance(grid_search_sizes, request):
    if request.config.getoption("--skip-perf", default=False):
        pytest.skip("Skipping performance test")
    data = grid_search_sizes
    args = (
        data["beta_hat"],
        data["sigma"],
        data["A"],
        data["d"],
        data["theta_grid"],
        data["n_pre_periods"],
        data["post_period_index"],
        data["alpha"],
    )
    kwargs = {"test_fn": arp_no_nuisance._test_in_identified_set}

    warm_up_iterations = 5 if data["grid_size"] <= 30 else 2
    for _ in range(warm_up_iterations):
        arp_no_nuisance._test_over_theta_grid(*args, **kwargs)

    time_numba = time_function(arp_no_nuisance._test_over_theta_grid, *args, **kwargs)
    time_original = time_function(_test_over_theta_grid_py, *args, **kwargs)

    if data["grid_size"] == 30:
        margin = 3.5
    elif data["grid_size"] == 50:
        margin = 2.0
    elif data["grid_size"] == 100:
        margin = 1.5
    else:
        margin = 1.2

    if data["grid_size"] >= 30:
        assert time_original * margin > time_numba, (
            f"Numba version should not be significantly slower for grid_size={data['grid_size']}"
        )
