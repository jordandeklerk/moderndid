import warnings

import numpy as np
import polars as pl
import pytest

from moderndid.didtriple.utils import (
    check_overlap_and_warn,
    detect_multiple_periods,
    detect_rcs_mode,
    get_comparison_description,
    get_covariate_names,
)


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("~ x1 + x2 + x3", ["x1", "x2", "x3"]),
        ("~ x1 + x2", ["x1", "x2"]),
        ("  ~ x1 + x2  ", ["x1", "x2"]),
        (None, None),
        ("~1", None),
    ],
)
def test_get_covariate_names(formula, expected):
    assert get_covariate_names(formula) == expected


@pytest.mark.parametrize(
    "times,groups,expected",
    [
        ([1, 1, 2, 2], [0, 3, 0, 3], False),
        ([1, 2, 3, 4, 5], [0, 0, 3, 3, 4], True),
        ([1, 1, 1, 2, 2, 2], [0, 3, 4, 0, 3, 4], True),
    ],
)
def test_detect_multiple_periods(times, groups, expected):
    df = pl.DataFrame(
        {
            "time": times,
            "group": groups,
            "y": [1.0] * len(times),
        }
    )
    assert detect_multiple_periods(df, "time", "group") == expected


def test_detect_multiple_periods_with_inf_groups():
    df = pl.DataFrame(
        {
            "time": [1, 1, 2, 2],
            "group": [np.inf, 3, np.inf, 3],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert detect_multiple_periods(df, "time", "group") is False


@pytest.mark.parametrize(
    "times,ids,panel,allow_unbalanced,expected",
    [
        ([1, 2], [1, 2], False, False, True),
        ([1, 2, 1, 2], [1, 1, 2, 2], True, False, False),
        ([1, 2, 1], [1, 1, 2], True, True, True),
    ],
)
def test_detect_rcs_mode(times, ids, panel, allow_unbalanced, expected):
    df = pl.DataFrame({"time": times, "id": ids})
    assert detect_rcs_mode(df, "time", "id", panel, allow_unbalanced) == expected


def test_detect_rcs_mode_no_idname():
    df = pl.DataFrame({"time": [1, 2]})
    assert detect_rcs_mode(df, "time", None, panel=True, allow_unbalanced_panel=False) is True


@pytest.mark.parametrize(
    "subgroup,expected",
    [
        (3, "Treated-Eligible vs Treated-Ineligible"),
        (2, "Treated-Eligible vs Eligible-Untreated"),
        (1, "Treated-Eligible vs Untreated-Ineligible"),
    ],
)
def test_get_comparison_description(subgroup, expected):
    assert get_comparison_description(subgroup) == expected


def test_check_overlap_and_warn():
    ps_good = np.array([0.3, 0.4, 0.5, 0.6, 0.7])
    ps_poor = np.array([0.0001] * 20 + [0.5] * 80)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_overlap_and_warn(ps_good, condition_subgroup=2)
        assert len(w) == 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_overlap_and_warn(ps_poor, condition_subgroup=2, threshold=1e-3, max_proportion=0.05)
        assert len(w) == 1
        assert "Poor propensity score overlap" in str(w[0].message)
