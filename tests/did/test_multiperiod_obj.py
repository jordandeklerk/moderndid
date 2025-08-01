"""Tests for multi-period (MP) result objects."""

import numpy as np
import pytest

from causaldid import (
    MPPretestResult,
    MPResult,
    format_mp_pretest_result,
    format_mp_result,
    mp,
    mp_pretest,
    summary_mp_pretest,
)


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


def test_mp_pretest_basic():
    cvm_boots = np.random.uniform(0, 5, 99)
    ks_boots = np.random.uniform(0, 3, 99)

    result = mp_pretest(
        cvm_stat=1.23,
        cvm_critval=1.46,
        cvm_pval=0.089,
        ks_stat=0.67,
        ks_critval=0.88,
        ks_pval=0.123,
        cvm_boots=cvm_boots,
        ks_boots=ks_boots,
        cluster_vars=["state", "year"],
        x_formula="~ X1 + X2 + X3",
    )

    assert isinstance(result, MPPretestResult)
    assert result.cvm_stat == 1.23
    assert result.cvm_critval == 1.46
    assert result.cvm_pval == 0.089
    assert result.ks_stat == 0.67
    assert result.ks_critval == 0.88
    assert result.ks_pval == 0.123
    assert len(result.cvm_boots) == 99
    assert len(result.ks_boots) == 99
    assert result.cluster_vars == ["state", "year"]
    assert result.x_formula == "~ X1 + X2 + X3"


def test_mp_pretest_minimal():
    result = mp_pretest(
        cvm_stat=2.5,
        cvm_critval=3.0,
        cvm_pval=0.043,
        ks_stat=1.2,
        ks_critval=1.5,
        ks_pval=0.085,
    )

    assert result.cvm_boots is None
    assert result.ks_boots is None
    assert result.cluster_vars is None
    assert result.x_formula is None


def test_mp_pretest_formatting():
    result = mp_pretest(
        cvm_stat=1.8765,
        cvm_critval=2.1234,
        cvm_pval=0.0567,
        ks_stat=0.9876,
        ks_critval=1.2345,
        ks_pval=0.0890,
        cluster_vars=["id"],
        x_formula="~ age + income",
    )

    formatted = format_mp_pretest_result(result)

    assert "Pre-test of Conditional Parallel Trends Assumption" in formatted
    assert "Cramer von Mises Test:" in formatted
    assert "Test Statistic: 1.8765" in formatted
    assert "Critical Value: 2.1234" in formatted
    assert "P-value       : 0.0567" in formatted
    assert "Kolmogorov-Smirnov Test:" in formatted
    assert "Test Statistic: 0.9876" in formatted
    assert "Critical Value: 1.2345" in formatted
    assert "P-value       : 0.0890" in formatted
    assert "Clustering on: id" in formatted
    assert "X formula: ~ age + income" in formatted


def test_mp_pretest_summary():
    result = mp_pretest(
        cvm_stat=1.0,
        cvm_critval=1.5,
        cvm_pval=0.15,
        ks_stat=0.8,
        ks_critval=1.0,
        ks_pval=0.20,
    )

    summary_output = summary_mp_pretest(result)
    assert isinstance(summary_output, str)
    assert "Cramer von Mises Test:" in summary_output
    assert "Kolmogorov-Smirnov Test:" in summary_output


def test_mp_pretest_display_methods():
    result = mp_pretest(
        cvm_stat=1.5,
        cvm_critval=2.0,
        cvm_pval=0.10,
        ks_stat=0.9,
        ks_critval=1.2,
        ks_pval=0.15,
    )

    str_output = str(result)
    repr_output = repr(result)

    assert isinstance(str_output, str)
    assert isinstance(repr_output, str)
    assert len(str_output) > 0
    assert len(repr_output) > 0
    assert "Pre-test of Conditional Parallel Trends Assumption" in str_output
