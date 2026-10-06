"""Shared configuration and fixtures."""

import os

import numpy as np
import pytest

_FAST_TEST_CONFIG = {
    "grid_points_small": 6,
    "grid_points_medium": 10,
    "grid_points_large": 16,
    "n_small": 4,
    "n_medium": 8,
    "n_large": 16,
    "n_sim_small": 8,
    "n_sim_medium": 16,
    "n_sim_large": 32,
    "skip_expensive_params": True,
}


@pytest.fixture
def fast_config():
    """Return configuration for fast test runs."""
    cfg = dict(_FAST_TEST_CONFIG)
    if os.environ.get("MODERNDID_RUN_FULL_TESTS"):
        cfg["skip_expensive_params"] = False
    return cfg


@pytest.fixture
def rm_cases():
    """Return event studies with three pre-periods whose largest first difference sits in different places."""
    sd = np.array([0.004, 0.004, 0.004, 0.005, 0.006])
    sigma = (np.full((5, 5), 0.2) + 0.8 * np.eye(5)) * np.outer(sd, sd)
    return {
        "sigma": sigma,
        "A": np.array([-0.01, -0.005, -0.02, 0.05, 0.06]),
        "B": np.array([-0.03, -0.02, -0.005, 0.05, 0.06]),
        "C": np.array([-0.05, -0.005, -0.02, 0.05, 0.06]),
    }


@pytest.fixture
def one_post_event_study():
    """Return an event study with four pre-periods and a single post-period."""
    sd = np.array([0.005, 0.004, 0.004, 0.003, 0.006])
    sigma = (np.full((5, 5), 0.3) + 0.7 * np.eye(5)) * np.outer(sd, sd)
    return {"betahat": np.array([-0.0125, -0.0067, -0.0038, -0.0049, 0.0453]), "sigma": sigma}


@pytest.fixture
def flci_event_study():
    """Return an event study with three pre-periods and two post-periods for fixed-length interval checks."""
    sd = 0.005 * (1 + 0.1 * np.arange(5))
    sigma = (np.full((5, 5), 0.6) + 0.4 * np.eye(5)) * np.outer(sd, sd)
    return {"betahat": np.linspace(-0.01, 0.05, 5), "sigma": sigma, "post_period_weights": np.array([1.0, 0.0])}


@pytest.fixture
def singular_pre_period_event_studies():
    """Return event studies with four pre-periods whose pre-period covariance is singular in different ways."""
    sd = np.array([0.004, 0.004, 0.004, 0.005, 0.006, 0.007])
    sigma = (np.full((6, 6), 0.3) + 0.7 * np.eye(6)) * np.outer(sd, sd)
    betahat = np.array([-0.01, 0.0, -0.004, 0.002, 0.03, 0.04])

    zero_variance = sigma.copy()
    zero_variance[1, :] = 0.0
    zero_variance[:, 1] = 0.0

    duplicated = sigma.copy()
    duplicated[1, :] = duplicated[0, :]
    duplicated[:, 1] = duplicated[:, 0]

    scaled_duplicate = sigma.copy()
    scaled_duplicate[1, :] = 2 * scaled_duplicate[0, :]
    scaled_duplicate[:, 1] = 2 * scaled_duplicate[:, 0]
    scaled_duplicate[1, 1] = 4 * sigma[0, 0]

    second_difference = np.array([0.0, 1.0, -2.0, 1.0, 0.0, 0.0])
    projection = np.eye(6) - np.outer(second_difference, second_difference) / (second_difference @ second_difference)

    return {
        "zero_variance": (betahat, zero_variance),
        "duplicated": (np.array([-0.01, -0.01, -0.004, 0.002, 0.03, 0.04]), duplicated),
        "scaled_duplicate": (np.array([-0.01, -0.02, -0.004, 0.002, 0.03, 0.04]), scaled_duplicate),
        "zero_variance_second_difference": (betahat, projection @ sigma @ projection),
    }


@pytest.fixture
def rank_deficient_event_study():
    """Return an event study with four pre-periods and a rank three covariance matrix."""
    x = np.random.default_rng(1).normal(size=(3, 6)) * 0.004
    return np.linspace(-0.01, 0.04, 6), x.T @ x


@pytest.fixture(scope="session")
def use_fast_tests(request):
    """Check if fast tests are requested via command line or environment."""
    return request.config.getoption("--fast", default=False)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast", action="store_true", default=False, help="Run tests in fast mode with reduced iterations/samples"
    )
    parser.addoption("--skip-perf", action="store_true", default=False, help="Skip performance benchmarking tests")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "perf: marks performance benchmarking tests")
