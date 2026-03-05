"""Root test configuration — shared warning filters.

Only numerical RuntimeWarnings and third-party warnings are suppressed globally.
Library UserWarnings are intentionally NOT suppressed so they remain visible
during test runs. Use per-test ``@pytest.mark.filterwarnings`` for warnings
that are expected in a specific test.
"""


def pytest_configure(config):
    """Register global warning filters via pytest's own mechanism."""
    filters = [
        # Numerical warnings from scipy/numpy in edge-case data
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
