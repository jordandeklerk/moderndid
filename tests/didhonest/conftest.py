"""Shared configuration and fixtures."""

import os

import numpy as np
import pytest

from moderndid import AGGTEResult

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


@pytest.fixture
def one_pre_period_event_study():
    """Return an event study with one pre-period and two post-periods."""
    sigma = np.array([[0.0004, 0.0001, 0.0001], [0.0001, 0.0009, 0.0003], [0.0001, 0.0003, 0.0010]])
    return {"betahat": np.array([-0.03, 0.10, 0.12]), "sigma": sigma}


@pytest.fixture
def two_pre_period_event_study():
    """Return an event study with two pre-periods and two post-periods."""
    sigma = np.array(
        [
            [0.010, 0.004, 0.002, 0.001],
            [0.004, 0.010, 0.003, 0.002],
            [0.002, 0.003, 0.020, 0.008],
            [0.001, 0.002, 0.008, 0.020],
        ]
    )
    return {"betahat": np.array([0.05, -0.02, 0.30, 0.40]), "sigma": sigma}


@pytest.fixture
def monotone_event_study():
    """Return an event study with four increasing pre-periods whose second differences all equal -0.01."""
    sigma = np.eye(7) * 0.01
    np.fill_diagonal(sigma[1:], 0.005)
    np.fill_diagonal(sigma[:, 1:], 0.005)
    return {"betahat": np.array([-0.10, -0.06, -0.03, -0.01, 0.15, 0.08, 0.12]), "sigma": sigma}


@pytest.fixture
def linear_trend_event_study():
    """Return an event study whose coefficients follow a precise linear trend and show no effect."""
    event_times = np.array([-3, -2, -1, 1, 2, 3, 4])
    sigma = (np.full((7, 7), 0.2) + 0.8 * np.eye(7)) * 0.004**2
    return {"betahat": 0.1 * event_times, "sigma": sigma}


@pytest.fixture
def far_effect_event_study():
    """Return an event study with a linear pre-trend and effects about 50 standard errors from zero."""
    sigma = (np.full((5, 5), 0.3) + 0.7 * np.eye(5)) * 0.01**2
    return {"betahat": np.array([-0.03, -0.02, -0.01, 0.50, 0.55]), "sigma": sigma}


@pytest.fixture
def noisy_long_horizon_event_study():
    """Return a draw with three pre-periods and eight post-periods around zero with a standard deviation of 0.05."""
    sigma = (np.full((11, 11), 0.3) + 0.7 * np.eye(11)) * 0.05**2
    betahat = np.array([0.0195, 0.0107, 0.0154, 0.0043, 0.0243, -0.053, -0.041, 0.0165, 0.0395, -0.0119, -0.0248])
    return {"betahat": betahat, "sigma": sigma}


@pytest.fixture
def make_event_study():
    """Build a dynamic aggregation with a universal base period and random unit-level influence functions."""

    def build(event_times, att_by_event, n_units=200, seed=0):
        influence_func = np.random.default_rng(seed).normal(0, 0.3, (n_units, len(event_times)))
        influence_func[:, list(event_times).index(-1)] = 0.0
        return AGGTEResult(
            overall_att=float(np.mean(att_by_event[event_times >= 0])),
            overall_se=0.05,
            aggregation_type="dynamic",
            event_times=event_times,
            att_by_event=att_by_event,
            se_by_event=np.sqrt(np.sum(influence_func**2, axis=0)) / n_units,
            influence_func=influence_func,
            estimation_params={"base_period": "universal"},
        )

    return build


@pytest.fixture
def clustered_event_study():
    """Return a dynamic aggregation with ten units in five clusters and a zero reference-period column."""
    influence_func = np.array(
        [
            [0.4, 0.0, 1.2, 0.3],
            [-0.2, 0.0, 0.8, -0.5],
            [0.1, 0.0, -0.6, 0.9],
            [0.3, 0.0, -1.1, 0.2],
            [-0.5, 0.0, 0.4, -0.7],
            [0.2, 0.0, 0.9, 0.6],
            [-0.1, 0.0, -0.3, -0.4],
            [0.6, 0.0, -0.8, 0.1],
            [-0.4, 0.0, 0.2, -0.2],
            [-0.4, 0.0, -0.7, -0.3],
        ]
    )
    return AGGTEResult(
        overall_att=0.35,
        overall_se=0.1,
        aggregation_type="dynamic",
        event_times=np.array([-2, -1, 0, 1]),
        att_by_event=np.array([0.02, 0.0, 0.30, 0.40]),
        se_by_event=np.array([0.05, np.nan, 0.08, 0.09]),
        influence_func=influence_func,
        estimation_params={
            "base_period": "universal",
            "clustervars": ["state"],
            "cluster": np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        },
    )


@pytest.fixture(scope="session")
def use_fast_tests(request):
    """Check if fast tests are requested via command line or environment."""
    return request.config.getoption("--fast", default=False)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast", action="store_true", default=False, help="Run tests in fast mode with reduced iterations/samples"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
