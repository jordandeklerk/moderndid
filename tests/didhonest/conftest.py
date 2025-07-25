"""Shared configuration and fixtures."""

import pytest

FAST_TEST_CONFIG = {
    "grid_points_small": 20,
    "grid_points_medium": 30,
    "grid_points_large": 50,
    "n_small": 10,
    "n_medium": 20,
    "n_large": 50,
    "n_sim_small": 50,
    "n_sim_medium": 100,
    "n_sim_large": 200,
    "skip_expensive_params": True,
}


@pytest.fixture
def fast_config():
    """Return configuration for fast test runs."""
    return FAST_TEST_CONFIG


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
