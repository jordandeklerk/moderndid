"""Root test warnings configuration and shared fixtures."""

import sys

import numpy as np
import polars as pl
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


@pytest.fixture
def forbid_errstate(monkeypatch):
    """Make np.errstate raise so that a test fails when library code hides floating-point warnings."""

    def forbidden(**kwargs):
        raise AssertionError("np.errstate hides floating-point warnings")

    def activate():
        monkeypatch.setattr(np, "errstate", forbidden)

    return activate


@pytest.fixture
def without_formulaic(monkeypatch):
    """Hide formulaic the way an install without extras does."""
    monkeypatch.setitem(sys.modules, "formulaic", None)
    monkeypatch.setattr("moderndid.core.preprocess.transformers.formulaic", None, raising=False)


@pytest.fixture
def zero_scale_draws():
    """Draws of two columns whose second column moves in ten of 21 draws and has a zero interquartile range."""
    first = np.arange(-10.0, 11.0)
    return np.column_stack([first, np.where(np.abs(first) > 5, np.sign(first), 0.0)])


@pytest.fixture
def central_zero_scale_draws():
    """Draws of two columns whose second column moves in eight of the 11 central draws of the first.

    The first column has five far draws on each side of its center. The second column has a zero
    interquartile range. Dropping the draws that move it leaves mostly far draws of the first.
    """
    first = np.concatenate([np.arange(-50.0, -9.0, 10.0), np.arange(-5.0, 6.0), np.arange(10.0, 51.0, 10.0)])
    return np.column_stack([first, np.where((np.abs(first) <= 4) & (first != 0), np.sign(first), 0.0)])


@pytest.fixture
def mboot_draws(request, central_zero_scale_draws):
    """Draws of columns that test the bootstrap scale rules, chosen by the scenario name."""
    first = central_zero_scale_draws[:, 0]
    if request.param == "zero_scale_moved":
        return central_zero_scale_draws
    if request.param == "zero_scale_unmoved":
        return np.column_stack([first, np.zeros_like(first)])
    if request.param == "negligible_scale":
        return np.column_stack([first, 1e-9 * np.where(np.abs(first) == 10, 8 * first, first)])
    if request.param == "missing":
        return np.column_stack([first, np.full_like(first, np.nan)])
    if request.param == "below_pointwise":
        return np.linspace(-1.0, 1.0, 99)[:, None]
    return np.random.default_rng(3).standard_normal((199, 4))


@pytest.fixture
def inf_func_with_zero_scale_column():
    """Influence function whose middle column is zero except for two offsetting spikes."""
    inf_func = np.random.default_rng(11).standard_normal((200, 3))
    inf_func[:, 1] = 0.0
    inf_func[:2, 1] = [3.0, -3.0]
    return inf_func


@pytest.fixture
def offsetting_spike_weights():
    """Build Rademacher weights in which two units share a weight in 600 of 999 draws.

    The two units take opposite weights in the other 399 draws, half of them positive for the first unit.
    """

    def build(n_units, first, second):
        weights = np.where(np.random.default_rng(7).random((999, n_units)) < 0.5, 1.0, -1.0)
        weights[:600, second] = weights[:600, first]
        weights[600:, first] = np.where(np.arange(399) % 2 == 0, 1.0, -1.0)
        weights[600:, second] = -weights[600:, first]
        return weights

    return build


@pytest.fixture
def draws_from_weights():
    """Turn fixed multiplier weights into a draw function that averages the weighted influence function."""

    def build(weights):
        def draws(inf_func, biters, random_state=None):
            return weights @ inf_func / inf_func.shape[0]

        return draws

    return build


@pytest.fixture
def did_offsetting_cohort_data():
    """Panel whose controls share one outcome path and where cohort 2 has two units with opposite shocks."""
    rng = np.random.default_rng(5)
    records = []
    for cohort, first_unit, n_units in ((0, 0, 100), (3, 100, 50), (4, 150, 50), (5, 200, 50), (2, 250, 40)):
        for unit in range(first_unit, first_unit + n_units):
            level = rng.normal()
            for time in range(1, 7):
                y = level + time + float(cohort > 0 and time >= cohort)
                if cohort in (3, 4, 5):
                    y += rng.normal(0, 0.5)
                if unit in (250, 251) and time >= 2:
                    y += (0.4 if unit == 250 else -0.4) * time
                records.append({"id": unit, "t": time, "y": y, "g": cohort})
    return pl.DataFrame(records)
