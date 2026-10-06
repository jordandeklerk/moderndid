"""Root test warnings configuration and shared fixtures."""

import numpy as np
import pytest


# Only numerical RuntimeWarnings and third-party warnings are suppressed globally.
# Library UserWarnings are intentionally NOT suppressed so they remain visible
# during test runs. Use per-test ``@pytest.mark.filterwarnings`` for warnings
# that are expected in a specific test.
def pytest_configure(config):
    """Register global warning filters via pytest's own mechanism."""
    filters = [
        # Numerical RuntimeWarnings in edge-case data
        "ignore:overflow encountered:RuntimeWarning",
        "ignore:invalid value encountered:RuntimeWarning",
        "ignore:divide by zero encountered:RuntimeWarning",
        "ignore:Mean of empty slice:RuntimeWarning",
        # Third-party warnings we cannot control
        "ignore:Perfect separation.*:statsmodels.tools.sm_exceptions.PerfectSeparationWarning",
        "ignore:Solution may be inaccurate.*:UserWarning",
        "ignore:np.dot.*is faster on contiguous arrays.*:numba.core.errors.NumbaPerformanceWarning",
        "ignore:.*pl.count.*is deprecated.*:DeprecationWarning",
    ]
    for f in filters:
        config.addinivalue_line("filterwarnings", f)


class FixedDraws(np.random.Generator):
    """Random generator whose integer draws come from a preset list of cluster draws."""

    def __init__(self, draws):
        super().__init__(np.random.PCG64(0))
        self._draws = iter(draws)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """Return the next preset draw."""
        return np.asarray(next(self._draws), dtype=np.int64)


@pytest.fixture
def fixed_draws():
    """Generator class that hands the bootstrap preset cluster draws."""
    return FixedDraws
