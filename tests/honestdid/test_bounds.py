"""Tests for the bounds module."""

import numpy as np
import pytest

from pydid.honestdid import (
    compute_delta_sd_lowerbound_m,
    compute_delta_sd_upperbound_m,
    create_pre_period_constraint_matrix,
    create_second_difference_matrix,
)


def test_compute_delta_sd_upperbound_m():
    betahat = np.array([0.1, 0.2, 0.4, 0.8])
    sigma = np.eye(4) * 0.01
    num_pre_periods = 4

    result = compute_delta_sd_upperbound_m(betahat, sigma, num_pre_periods, alpha=0.05)

    assert result > 0
    assert result > 0.1


def test_compute_delta_sd_upperbound_m_edge_case():
    betahat = np.array([0.1, 0.2, 0.35])
    sigma = np.eye(3) * 0.01
    num_pre_periods = 3

    result = compute_delta_sd_upperbound_m(betahat, sigma, num_pre_periods)
    assert result > 0


def test_compute_delta_sd_upperbound_m_invalid_periods():
    betahat = np.array([0.1, 0.2])
    sigma = np.eye(2) * 0.01
    num_pre_periods = 2

    with pytest.raises(ValueError, match="Cannot estimate M"):
        compute_delta_sd_upperbound_m(betahat, sigma, num_pre_periods)


def test_compute_delta_sd_lowerbound_m():
    np.random.seed(42)
    betahat = np.array([0.1, 0.15, 0.22, 0.31])
    sigma = np.eye(4) * 0.01
    num_pre_periods = 4

    result = compute_delta_sd_lowerbound_m(betahat, sigma, num_pre_periods, alpha=0.05, grid_ub=1.0, grid_points=100)

    assert result >= 0
    assert result < 0.5


def test_compute_delta_sd_lowerbound_m_no_grid_ub():
    betahat = np.array([0.1, 0.2, 0.3])
    sigma = np.eye(3) * 0.04
    num_pre_periods = 3

    result = compute_delta_sd_lowerbound_m(betahat, sigma, num_pre_periods, alpha=0.05, grid_points=50)

    assert result >= 0


def test_create_second_difference_matrix():
    num_pre_periods = 4
    num_post_periods = 3

    A = create_second_difference_matrix(num_pre_periods, num_post_periods)

    assert A.shape == (3, 7)

    assert np.array_equal(A[0, :3], [1, -2, 1])
    assert np.array_equal(A[1, 1:4], [1, -2, 1])
    assert np.array_equal(A[2, 4:7], [1, -2, 1])


def test_create_second_difference_matrix_pre_only():
    num_pre_periods = 3
    num_post_periods = 0

    A = create_second_difference_matrix(num_pre_periods, num_post_periods)

    assert A.shape == (1, 3)
    assert np.array_equal(A[0, :], [1, -2, 1])


def test_create_second_difference_matrix_insufficient_periods():
    num_pre_periods = 1
    num_post_periods = 1

    A = create_second_difference_matrix(num_pre_periods, num_post_periods)

    assert A.shape == (0, 2)


def test_create_pre_period_constraint_matrix():
    num_pre_periods = 4

    A, d = create_pre_period_constraint_matrix(num_pre_periods)

    assert A.shape == (6, 4)
    assert len(d) == 6

    assert np.all(d == 1)
    assert np.array_equal(A[:3, :], -A[3:, :])


def test_create_pre_period_constraint_matrix_minimum():
    num_pre_periods = 2

    A, d = create_pre_period_constraint_matrix(num_pre_periods)

    assert A.shape == (0, 2)
    assert len(d) == 0


def test_create_pre_period_constraint_matrix_invalid():
    with pytest.raises(ValueError, match="Cannot estimate M"):
        create_pre_period_constraint_matrix(1)


def test_integration_upper_lower_bounds():
    np.random.seed(123)
    betahat = np.random.normal(0, 0.1, 5)
    sigma = np.eye(5) * 0.01
    num_pre_periods = 5

    upper = compute_delta_sd_upperbound_m(betahat, sigma, num_pre_periods)
    lower = compute_delta_sd_lowerbound_m(betahat, sigma, num_pre_periods, grid_ub=upper * 2, grid_points=50)

    assert upper >= lower
