"""Tests for multi-period (MP) result objects."""

import re

import numpy as np
import pytest

from moderndid import (
    MPPretestResult,
    MPResult,
    format_mp_pretest_result,
    format_mp_result,
    mp,
    mp_pretest,
    summary_mp_pretest,
)


def test_mp_stores_its_arguments_with_sequences_as_arrays():
    vcov = np.diag([0.1, 0.2, 0.3, 0.4])
    influence_func = np.random.default_rng(0).normal(0, 0.1, (100, 4))
    cohorts = np.repeat([2000.0, 2001.0, np.inf], [30, 30, 40])
    weights = np.linspace(0.5, 1.5, 100)

    result = mp(
        groups=[2000, 2000, 2001, 2001],
        times=[2001, 2002, 2002, 2003],
        att_gt=[0.8, 1.2, 1.5, 1.1],
        vcov_analytical=vcov,
        se_gt=[0.2, 0.25, 0.3, 0.22],
        critical_value=2.24,
        influence_func=influence_func,
        n_units=100,
        wald_stat=5.23,
        wald_pvalue=0.073,
        alpha=0.1,
        G=cohorts,
        weights_ind=weights,
    )

    assert isinstance(result, MPResult)
    assert all(isinstance(array, np.ndarray) for array in (result.groups, result.times, result.att_gt, result.se_gt))
    np.testing.assert_array_equal(result.groups, [2000, 2000, 2001, 2001])
    np.testing.assert_array_equal(result.times, [2001, 2002, 2002, 2003])
    np.testing.assert_array_equal(result.att_gt, [0.8, 1.2, 1.5, 1.1])
    np.testing.assert_array_equal(result.se_gt, [0.2, 0.25, 0.3, 0.22])
    np.testing.assert_array_equal(result.vcov_analytical, vcov)
    np.testing.assert_array_equal(result.influence_func, influence_func)
    np.testing.assert_array_equal(result.G, cohorts)
    np.testing.assert_array_equal(result.weights_ind, weights)
    assert (result.critical_value, result.n_units, result.wald_stat, result.wald_pvalue, result.alpha) == (
        2.24,
        100,
        5.23,
        0.073,
        0.1,
    )
    assert (result.aggregate_effects, result.influence_moments) == (None, None)


@pytest.mark.parametrize(
    ("n_times", "n_att", "n_se", "message"),
    [
        (3, 2, 2, "groups and times must have the same length."),
        (2, 3, 2, "att_gt must have same length as groups and times."),
        (2, 2, 3, "se_gt must have same length as groups and times."),
    ],
)
def test_mp_rejects_arrays_of_different_lengths(n_times, n_att, n_se, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        mp(
            groups=np.array([2000, 2001]),
            times=np.arange(2001, 2001 + n_times),
            att_gt=np.ones(n_att),
            vcov_analytical=np.eye(2),
            se_gt=np.full(n_se, 0.2),
            critical_value=1.96,
            influence_func=np.zeros((100, 2)),
        )


def test_mp_stores_estimation_params_and_defaults_to_an_empty_dict():
    spec = {
        "groups": np.array([2000, 2000, 2001, 2001]),
        "times": np.array([2001, 2002, 2002, 2003]),
        "att_gt": np.array([0.8, 1.2, 1.5, 1.1]),
        "vcov_analytical": np.eye(4) * 0.1,
        "se_gt": np.array([0.2, 0.25, 0.3, 0.22]),
        "critical_value": 2.24,
        "influence_func": np.zeros((50, 4)),
    }
    estimation_params = {
        "control_group": "nevertreated",
        "anticipation_periods": 0,
        "estimation_method": "dr",
        "bootstrap": True,
        "uniform_bands": True,
    }

    result = mp(**spec, estimation_params=estimation_params)
    default = mp(**spec)

    assert result.estimation_params == estimation_params
    assert default.estimation_params == {}


def test_mp_formatting_reports_each_cell_with_its_interval_and_the_fit_settings():
    result = mp(
        groups=np.array([2000, 2000, 2001, 2001]),
        times=np.array([2001, 2002, 2001, 2002]),
        att_gt=np.array([1.5, -0.3, -2.0, 2.1]),
        vcov_analytical=np.eye(4),
        se_gt=np.array([0.5, 0.4, 0.4, np.nan]),
        critical_value=2.5,
        influence_func=np.zeros((100, 4)),
        wald_pvalue=0.23,
        alpha=0.1,
        estimation_params={
            "control_group": "notyettreated",
            "anticipation_periods": 1,
            "estimation_method": "ipw",
        },
    )

    lines = format_mp_result(result).splitlines()
    rows = [[cell.strip() for cell in line.split("│")[1:-1]] for line in lines if line.startswith("│")]

    assert lines[1] == " Group-Time Average Treatment Effects"
    assert rows == [
        ["Group", "Time", "ATT(g,t)", "Std. Error", "[90% Pointwise Conf. Band]"],
        ["2000", "2001", "1.5000", "0.5000", "[ 0.2500,  2.7500] *"],
        ["2000", "2002", "-0.3000", "0.4000", "[-1.3000,  0.7000]"],
        ["2001", "2001", "-2.0000", "0.4000", "[-3.0000, -1.0000] *"],
        ["2001", "2002", "2.1000", "NA", "NA"],
    ]
    assert " P-value for pre-test of parallel trends assumption:  0.2300" in lines
    assert " Control Group:  Not Yet Treated" in lines
    assert " Anticipation Periods:  1" in lines
    assert " Estimation Method:  Inverse Probability Weighting" in lines
    assert lines[-4:-2] == [" Significance level: 0.1", " Analytical standard errors"]


@pytest.mark.parametrize(
    ("bootstrap", "uniform_bands", "band"),
    [
        (True, True, "[95% Simult. Conf. Band]"),
        (True, False, "[95% Pointwise Conf. Band]"),
        (False, True, "[95% Pointwise Conf. Band]"),
        (False, False, "[95% Pointwise Conf. Band]"),
    ],
)
def test_mp_formatting_calls_only_a_bootstrap_uniform_band_simultaneous(bootstrap, uniform_bands, band):
    result = mp(
        groups=np.array([2000, 2001]),
        times=np.array([2001, 2002]),
        att_gt=np.array([1.5, 2.1]),
        vcov_analytical=np.eye(2),
        se_gt=np.array([0.5, 0.6]),
        critical_value=2.3,
        influence_func=np.zeros((100, 2)),
        estimation_params={"bootstrap": bootstrap, "uniform_bands": uniform_bands},
    )

    header = next(line for line in format_mp_result(result).splitlines() if line.startswith("│ Group"))

    assert header.split("│")[-2].strip() == band


@pytest.mark.parametrize(
    ("estimation_params", "lines"),
    [
        (
            {"control_group": "nevertreated", "estimation_method": "dr"},
            [" Control Group:  Never Treated", " Estimation Method:  Doubly Robust"],
        ),
        (
            {"control_group": "notyettreated", "estimation_method": "reg"},
            [" Control Group:  Not Yet Treated", " Estimation Method:  Outcome Regression"],
        ),
        (
            {"control_group": "custom", "estimation_method": "custom"},
            [" Control Group:  custom", " Estimation Method:  custom"],
        ),
    ],
)
def test_mp_formatting_names_the_control_group_and_estimation_method(estimation_params, lines):
    result = mp(
        groups=np.array([2000]),
        times=np.array([2001]),
        att_gt=np.array([1.0]),
        vcov_analytical=np.array([[0.09]]),
        se_gt=np.array([0.3]),
        critical_value=1.96,
        influence_func=np.zeros((100, 1)),
        estimation_params=estimation_params,
    )

    report = format_mp_result(result).splitlines()

    assert [line for line in report if line.startswith((" Control Group", " Estimation Method"))] == lines


@pytest.mark.parametrize(
    ("estimation_params", "line"),
    [
        ({"bootstrap": False}, " Analytical standard errors"),
        ({"bootstrap": True}, " Bootstrap standard errors"),
        (
            {"bootstrap": True, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)},
            " Bootstrap standard errors clustered by state",
        ),
        (
            {"bootstrap": False, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)},
            " Analytical standard errors clustered by state",
        ),
        ({"bootstrap": True, "clustervars": ["copy"], "cluster": np.arange(100)}, " Bootstrap standard errors"),
    ],
)
def test_mp_formatting_names_the_clusters_of_the_standard_errors(estimation_params, line):
    result = mp(
        groups=np.array([2000, 2000, 2001]),
        times=np.array([2001, 2002, 2002]),
        att_gt=np.array([1.5, -0.3, 2.1]),
        vcov_analytical=np.eye(3),
        se_gt=np.array([0.5, 0.4, 0.6]),
        critical_value=1.96,
        influence_func=np.zeros((100, 3)),
        estimation_params=estimation_params,
    )

    lines = format_mp_result(result).splitlines()

    assert line in lines
    assert lines[lines.index(line) - 1] == " Significance level: 0.05"


@pytest.mark.parametrize(
    ("estimation_params", "wald_pvalue", "explained"),
    [
        ({"bootstrap": True, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)}, None, True),
        ({"bootstrap": False, "clustervars": ["state"], "cluster": np.repeat(np.arange(50), 2)}, None, True),
        ({"bootstrap": False}, None, False),
        ({"bootstrap": True, "clustervars": ["copy"], "cluster": np.arange(100)}, 0.23, False),
    ],
    ids=["bootstrap_clusters", "analytic_clusters", "no_pre_test", "unit_level_clusters"],
)
def test_mp_formatting_explains_the_missing_pre_test_of_a_clustered_fit(estimation_params, wald_pvalue, explained):
    result = mp(
        groups=np.array([2000, 2000, 2001]),
        times=np.array([2001, 2002, 2002]),
        att_gt=np.array([1.5, -0.3, 2.1]),
        vcov_analytical=np.eye(3),
        se_gt=np.array([0.5, 0.4, 0.6]),
        critical_value=1.96,
        influence_func=np.zeros((100, 3)),
        wald_pvalue=wald_pvalue,
        estimation_params=estimation_params,
    )
    explanation = [
        " Since a Wald test on a cluster-robust covariance rejects too often at",
        " realistic cluster counts, no pre-test of parallel trends is reported.",
        " Use the bootstrap's uniform bands over the pre-treatment cells instead.",
    ]

    lines = format_mp_result(result).splitlines()

    assert all(line in lines for line in explanation) == explained
    assert any("P-value for pre-test" in line for line in lines) == (wald_pvalue is not None)


@pytest.mark.parametrize("method", [str, repr])
def test_mp_str_and_repr_show_the_formatted_report(method):
    result = mp(
        groups=np.array([2000]),
        times=np.array([2001]),
        att_gt=np.array([1.0]),
        vcov_analytical=np.array([[0.09]]),
        se_gt=np.array([0.3]),
        critical_value=1.96,
        influence_func=np.zeros((100, 1)),
    )

    output = method(result)

    assert output == format_mp_result(result)


def test_mp_pretest_stores_its_arguments_with_bootstrap_draws_as_arrays():
    result = mp_pretest(
        cvm_stat=1.23,
        cvm_critval=1.46,
        cvm_pval=0.089,
        ks_stat=0.67,
        ks_critval=0.88,
        ks_pval=0.123,
        cvm_boots=[0.4, 1.1, 2.7],
        ks_boots=[0.2, 0.5, 0.9],
        cluster_vars=["state", "year"],
        x_formula="~ X1 + X2 + X3",
    )

    assert isinstance(result, MPPretestResult)
    assert (result.cvm_stat, result.cvm_critval, result.cvm_pval) == (1.23, 1.46, 0.089)
    assert (result.ks_stat, result.ks_critval, result.ks_pval) == (0.67, 0.88, 0.123)
    assert isinstance(result.cvm_boots, np.ndarray)
    assert isinstance(result.ks_boots, np.ndarray)
    np.testing.assert_array_equal(result.cvm_boots, [0.4, 1.1, 2.7])
    np.testing.assert_array_equal(result.ks_boots, [0.2, 0.5, 0.9])
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


def test_mp_pretest_formatting_reports_both_tests_in_order():
    result = mp_pretest(
        cvm_stat=1.87654,
        cvm_critval=2.1234,
        cvm_pval=0.05671,
        ks_stat=0.9876,
        ks_critval=1.23449,
        ks_pval=0.08896,
        cluster_vars=["id", "state"],
        x_formula="~ age + income",
    )

    lines = format_mp_pretest_result(result).splitlines()

    assert lines[1] == " Pre-test of Conditional Parallel Trends Assumption"
    assert lines[3:] == [
        "",
        " Cramer von Mises Test:",
        "   Test Statistic: 1.8765",
        "   Critical Value: 2.1234",
        "   P-value       : 0.0567",
        "",
        " Kolmogorov-Smirnov Test:",
        "   Test Statistic: 0.9876",
        "   Critical Value: 1.2345",
        "   P-value       : 0.0890",
        "",
        " Clustering on: id, state",
        " X formula: ~ age + income",
        "=" * 78,
    ]


def test_mp_pretest_formatting_leaves_out_missing_clusters_and_formula():
    result = mp_pretest(
        cvm_stat=1.0,
        cvm_critval=1.5,
        cvm_pval=0.15,
        ks_stat=0.8,
        ks_critval=1.0,
        ks_pval=0.20,
    )

    lines = format_mp_pretest_result(result).splitlines()

    assert lines[-3:] == ["   Critical Value: 1.0000", "   P-value       : 0.2000", "=" * 78]


@pytest.mark.parametrize("method", [str, repr, summary_mp_pretest])
def test_mp_pretest_str_repr_and_summary_show_the_formatted_report(method):
    result = mp_pretest(
        cvm_stat=1.5,
        cvm_critval=2.0,
        cvm_pval=0.10,
        ks_stat=0.9,
        ks_critval=1.2,
        ks_pval=0.15,
    )

    output = method(result)

    assert output == format_mp_pretest_result(result)
