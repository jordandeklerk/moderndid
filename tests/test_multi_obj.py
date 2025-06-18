"""Tests for multi-period (MP) result objects."""

import numpy as np
import pytest

from pydid import MPResult, format_mp_result, mp


def test_mp_basic():
    n_gt = 10
    groups = np.array([2000, 2000, 2000, 2001, 2001, 2001, 2002, 2002, 2002, 2003])
    times = np.array([2001, 2002, 2003, 2002, 2003, 2004, 2003, 2004, 2005, 2004])
    att_gt = np.random.normal(1.0, 0.5, n_gt)
    se_gt = np.random.uniform(0.1, 0.3, n_gt)
    vcov = np.eye(n_gt) * 0.1
    influence_func = np.random.normal(0, 0.1, (100, n_gt))

    result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=influence_func,
        n_units=100,
        wald_stat=5.23,
        wald_pvalue=0.073,
        alpha=0.05,
    )

    assert isinstance(result, MPResult)
    assert len(result.groups) == n_gt
    assert len(result.times) == n_gt
    assert len(result.att_gt) == n_gt
    assert len(result.se_gt) == n_gt
    assert result.critical_value == 1.96
    assert result.n_units == 100
    assert result.wald_stat == 5.23
    assert result.wald_pvalue == 0.073
    assert result.alpha == 0.05


def test_mp_validation():
    groups = np.array([2000, 2001])
    times = np.array([2001, 2002, 2003])
    att_gt = np.array([1.0, 1.5])
    se_gt = np.array([0.2, 0.3])
    vcov = np.eye(2)
    influence_func = np.random.normal(0, 0.1, (100, 2))

    with pytest.raises(ValueError, match="groups and times must have the same length"):
        mp(
            groups=groups,
            times=times,
            att_gt=att_gt,
            vcov_analytical=vcov,
            se_gt=se_gt,
            critical_value=1.96,
            influence_func=influence_func,
        )


def test_mp_with_estimation_params():
    n_gt = 4
    groups = np.array([2000, 2000, 2001, 2001])
    times = np.array([2001, 2002, 2002, 2003])
    att_gt = np.array([0.8, 1.2, 1.5, 1.1])
    se_gt = np.array([0.2, 0.25, 0.3, 0.22])
    vcov = np.eye(n_gt) * 0.1
    influence_func = np.random.normal(0, 0.1, (50, n_gt))

    estimation_params = {
        "control_group": "nevertreated",
        "anticipation_periods": 0,
        "estimation_method": "dr",
        "bootstrap": True,
        "uniform_bands": True,
        "call_info": "att_gt(y ~ x | g, data=panel_data)",
    }

    result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=2.24,
        influence_func=influence_func,
        estimation_params=estimation_params,
    )

    assert result.estimation_params["control_group"] == "nevertreated"
    assert result.estimation_params["bootstrap"] is True
    assert result.estimation_params["uniform_bands"] is True


def test_mp_formatting():
    groups = np.array([2000, 2000, 2001])
    times = np.array([2001, 2002, 2002])
    att_gt = np.array([1.5, -0.3, 2.1])
    se_gt = np.array([0.5, 0.4, 0.6])
    vcov = np.eye(3)
    influence_func = np.random.normal(0, 0.1, (100, 3))

    result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=influence_func,
        wald_pvalue=0.23,
        estimation_params={
            "control_group": "notyettreated",
            "anticipation_periods": 1,
            "estimation_method": "ipw",
        },
    )

    formatted = format_mp_result(result)

    assert "Group-Time Average Treatment Effects:" in formatted
    assert "Group" in formatted
    assert "Time" in formatted
    assert "ATT(g,t)" in formatted
    assert "Std. Error" in formatted
    assert "Conf. Band]" in formatted
    assert "Control Group:  Not Yet Treated" in formatted
    assert "Anticipation Periods:  1" in formatted
    assert "Estimation Method:  Inverse Probability Weighting" in formatted
    assert "P-value for pre-test of parallel trends assumption:  0.2300" in formatted


def test_mp_display_methods():
    groups = np.array([2000])
    times = np.array([2001])
    att_gt = np.array([1.0])
    se_gt = np.array([0.3])

    result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=np.array([[0.09]]),
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=np.random.normal(0, 0.1, (100, 1)),
    )

    str_output = str(result)
    repr_output = repr(result)

    assert isinstance(str_output, str)
    assert isinstance(repr_output, str)
    assert len(str_output) > 0
    assert len(repr_output) > 0
