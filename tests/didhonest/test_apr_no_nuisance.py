# pylint: disable=redefined-outer-name, unused-argument
"""Tests for APR CI with no nuisance parameters."""

import numpy as np
import pytest

from moderndid.didhonest import arp_no_nuisance
from moderndid.didhonest.arp_no_nuisance import APRCIResult, compute_arp_ci
from moderndid.didhonest.bounds import create_second_difference_matrix


@pytest.fixture
def event_study_data():
    np.random.seed(42)
    n_pre = 4
    n_post = 4

    true_effect = 0.5
    pre_trend = 0.05
    pre_coefs = np.arange(n_pre) * pre_trend + np.random.normal(0, 0.1, n_pre)
    post_coefs = true_effect + np.random.normal(0, 0.15, n_post)
    beta_hat = np.concatenate([pre_coefs, post_coefs])

    sigma = np.eye(n_pre + n_post) * 0.05
    for i in range(len(sigma)):
        for j in range(len(sigma)):
            if i != j:
                sigma[i, j] = 0.01 * np.exp(-abs(i - j))

    sigma = (sigma + sigma.T) / 2
    sigma += np.eye(len(sigma)) * 0.01

    return beta_hat, sigma, n_pre, n_post


@pytest.fixture
def constraint_matrices(event_study_data):
    _, _, n_pre, n_post = event_study_data
    A_sd = create_second_difference_matrix(n_pre, n_post)
    A = np.vstack([A_sd, -A_sd])
    d = np.ones(A.shape[0])
    return A, d


def test_compute_arp_ci_basic(event_study_data, constraint_matrices):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_index=1,
        alpha=0.05,
        grid_points=100,
    )

    assert isinstance(result, APRCIResult)
    assert result.status == "success"
    assert result.ci_lower < result.ci_upper
    assert result.ci_length > 0
    assert len(result.theta_grid) == 100
    assert len(result.accept_grid) == 100
    assert np.any(result.accept_grid)


@pytest.mark.parametrize("post_idx", [1, 2, 3, 4])
def test_compute_arp_ci_different_post_periods(event_study_data, constraint_matrices, post_idx):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_index=post_idx,
        alpha=0.05,
        grid_points=50,
    )
    assert result.status == "success"
    assert result.ci_lower < result.ci_upper


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
def test_compute_arp_ci_alpha_levels(event_study_data, constraint_matrices, alpha):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        post_period_index=1,
        alpha=alpha,
        grid_points=50,
    )

    assert result.status == "success"
    assert result.ci_length > 0


def test_ci_length_ordering(event_study_data):
    beta_hat, sigma, n_pre, n_post = event_study_data

    A_sd = create_second_difference_matrix(n_pre, n_post)
    A = np.vstack([A_sd, -A_sd])
    d = np.ones(A.shape[0]) * 2.0

    alphas = [0.2, 0.1, 0.05]
    ci_lengths = []

    for alpha in alphas:
        result = compute_arp_ci(
            beta_hat=beta_hat,
            sigma=sigma,
            A=A,
            d=d,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            post_period_index=1,
            alpha=alpha,
            grid_points=300,
            grid_lb=-3.0,
            grid_ub=4.0,
        )
        ci_lengths.append(result.ci_length)

    assert ci_lengths[0] <= ci_lengths[1] <= ci_lengths[2]


def test_test_in_identified_set():
    sigma = np.array([[0.1, 0.02], [0.02, 0.1]])
    A = np.array([[1, 0], [0, 1]])
    d = np.array([0.5, 0.5])

    y_in = np.array([0.1, 0.2])
    in_set = arp_no_nuisance.test_in_identified_set(y_in, sigma, A, d, alpha=0.05)
    assert in_set

    A_full = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    d_full = np.array([1.0, 1.0, 1.0, 1.0])

    y_clearly_out = np.array([2.0, 2.0])
    not_in_set = arp_no_nuisance.test_in_identified_set(y_clearly_out, sigma, A_full, d_full, alpha=0.10)
    assert not not_in_set

    y_well_within = np.array([0.0, 0.0])
    well_within_set = arp_no_nuisance.test_in_identified_set(y_well_within, sigma, A_full, d_full, alpha=0.05)
    assert well_within_set


@pytest.mark.parametrize(
    "error_case,kwargs,match",
    [
        ("wrong_beta_length", {"beta_hat_slice": slice(None, -1)}, "beta_hat length"),
        ("invalid_post_index", {"post_period_index": 0}, "post_period_index"),
        ("non_square_sigma", {"sigma_slice": (slice(None), slice(None, -1))}, "sigma must be square"),
    ],
)
def test_compute_arp_ci_invalid_inputs(event_study_data, constraint_matrices, error_case, kwargs, match):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    if "beta_hat_slice" in kwargs:
        beta_hat = beta_hat[kwargs["beta_hat_slice"]]
        kwargs.pop("beta_hat_slice")

    if "sigma_slice" in kwargs:
        sigma = sigma[kwargs["sigma_slice"]]
        kwargs.pop("sigma_slice")

    with pytest.raises(ValueError, match=match):
        compute_arp_ci(beta_hat=beta_hat, sigma=sigma, A=A, d=d, n_pre_periods=n_pre, n_post_periods=n_post, **kwargs)


def test_compute_arp_ci_return_length(event_study_data, constraint_matrices):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        alpha=0.05,
        grid_points=50,
        return_length=False,
    )

    length = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        alpha=0.05,
        grid_points=50,
        return_length=True,
    )

    assert isinstance(length, float)
    assert length == result.ci_length


def test_compute_arp_ci_empty_ci():
    np.random.seed(123)
    n_pre = 4
    n_post = 4

    beta_hat = np.array([0.0, 1.0, 2.0, 3.0, 0.5, 0.5, 0.5, 0.5])

    sigma = np.eye(n_pre + n_post) * 0.001

    A_sd = create_second_difference_matrix(n_pre, n_post)
    A = np.vstack([A_sd, -A_sd])
    d = np.ones(A.shape[0]) * 0.01

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        alpha=0.001,
        grid_points=50,
        grid_lb=-0.5,
        grid_ub=1.5,
    )

    if result.status == "success":
        assert result.ci_length > 0
    else:
        assert result.status == "empty_ci"
        assert np.isnan(result.ci_lower)
        assert np.isnan(result.ci_upper)
        assert np.isnan(result.ci_length)


@pytest.mark.parametrize(
    "grid_lb,grid_ub,grid_points",
    [
        (-1.0, 2.0, 50),
        (-2.0, 3.0, 100),
        (0.0, 1.0, 25),
    ],
)
def test_compute_arp_ci_custom_grid_bounds(event_study_data, constraint_matrices, grid_lb, grid_ub, grid_points):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    result = compute_arp_ci(
        beta_hat=beta_hat,
        sigma=sigma,
        A=A,
        d=d,
        n_pre_periods=n_pre,
        n_post_periods=n_post,
        grid_lb=grid_lb,
        grid_ub=grid_ub,
        grid_points=grid_points,
    )

    assert result.theta_grid[0] == grid_lb
    assert result.theta_grid[-1] == grid_ub
    assert len(result.theta_grid) == grid_points


def test_hybrid_flci_requires_params(event_study_data, constraint_matrices):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    with pytest.raises(ValueError, match="hybrid_kappa must be specified"):
        compute_arp_ci(
            beta_hat=beta_hat,
            sigma=sigma,
            A=A,
            d=d,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            hybrid_flag="FLCI",
        )

    with pytest.raises(ValueError, match="flci_halflength and flci_l must be specified"):
        compute_arp_ci(
            beta_hat=beta_hat,
            sigma=sigma,
            A=A,
            d=d,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            hybrid_flag="FLCI",
            hybrid_kappa=0.005,
        )


def test_hybrid_lf_requires_params(event_study_data, constraint_matrices):
    beta_hat, sigma, n_pre, n_post = event_study_data
    A, d = constraint_matrices

    with pytest.raises(ValueError, match="lf_cv must be specified"):
        compute_arp_ci(
            beta_hat=beta_hat,
            sigma=sigma,
            A=A,
            d=d,
            n_pre_periods=n_pre,
            n_post_periods=n_post,
            hybrid_flag="LF",
            hybrid_kappa=0.005,
        )
