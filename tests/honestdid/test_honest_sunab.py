# pylint: disable=redefined-outer-name,protected-access
"""Tests for the Honest Sun & Abraham estimator."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pydid import SunAbrahamCoefficients, extract_sunab_coefficients, sunab


@pytest.fixture
def sunab_panel_data():
    np.random.seed(42)

    n_units = 250
    periods = np.arange(2008, 2016)

    cohorts = [2010, 2012, 2014, 9999]
    units_per_cohort = n_units // len(cohorts)

    unit_ids = []
    cohort_values = []
    period_values = []

    for i, cohort in enumerate(cohorts):
        for unit in range(units_per_cohort):
            for period in periods:
                unit_ids.append(i * units_per_cohort + unit)
                cohort_values.append(cohort)
                period_values.append(period)

    df = pd.DataFrame({"unit": unit_ids, "cohort": cohort_values, "period": period_values})

    df["treated"] = (df["cohort"] != 9999) & (df["period"] >= df["cohort"])
    df["rel_period"] = np.where(df["cohort"] == 9999, -999, df["period"] - df["cohort"])

    np.random.seed(42)
    unit_fe = np.random.normal(0, 1, n_units)[df["unit"]]
    time_fe = 0.1 * (df["period"] - 2008)

    treatment_effect = np.zeros(len(df))
    for cohort in [2010, 2012, 2014]:
        cohort_mask = df["cohort"] == cohort
        for rel_time in range(-5, 8):
            mask = cohort_mask & (df["rel_period"] == rel_time)
            if rel_time >= 0:
                base_effect = 0.5 + 0.1 * (cohort - 2010) / 2
                treatment_effect[mask] = base_effect + 0.05 * rel_time

    df["y"] = unit_fe + time_fe + treatment_effect + np.random.normal(0, 0.5, len(df))

    return df


@pytest.fixture
def sunab_result_with_regression(sunab_panel_data):
    df = sunab_panel_data

    unit_dummies = pd.get_dummies(df["unit"], prefix="unit", drop_first=True)
    time_dummies = pd.get_dummies(df["period"], prefix="period", drop_first=True)
    covariates = np.column_stack([np.ones(len(df)), unit_dummies.values, time_dummies.values])

    result = sunab(
        cohort=df["cohort"].values,
        period=df["period"].values,
        outcome=df["y"].values,
        covariates=covariates,
        ref_period=-1,
        no_agg=True,
    )

    interactions = []
    interaction_names = []

    for cohort in df["cohort"].unique():
        if cohort == 9999:
            continue
        for period in sorted(df["rel_period"].unique()):
            if period in (-999, -1):
                continue
            mask = (df["cohort"] == cohort) & (df["rel_period"] == period)
            if mask.sum() > 0:
                interactions.append(mask.astype(float).values)
                interaction_names.append(f"sunab::{int(cohort)}::{int(period)}")

    X = np.column_stack([covariates, np.column_stack(interactions)])
    model = sm.OLS(df["y"].values, X)
    reg_result = model.fit()

    covariate_names = (
        ["const"]
        + [f"unit_{i}" for i in range(1, unit_dummies.shape[1] + 1)]
        + [f"period_{p}" for p in time_dummies.columns]
    )
    reg_result._results.params = pd.Series(reg_result.params, index=covariate_names + interaction_names)

    return result, reg_result


def test_extract_sunab_coefficients_integration(sunab_result_with_regression):
    _, reg_result = sunab_result_with_regression

    extracted = extract_sunab_coefficients(reg_result)

    assert isinstance(extracted, SunAbrahamCoefficients)
    assert isinstance(extracted.beta, np.ndarray)
    assert isinstance(extracted.sigma, np.ndarray)
    assert isinstance(extracted.event_times, np.ndarray)

    assert len(extracted.beta) == len(extracted.event_times)
    assert extracted.sigma.shape == (len(extracted.beta), len(extracted.beta))

    assert np.all(np.isfinite(extracted.beta))
    assert np.all(np.isfinite(extracted.sigma))
    assert np.allclose(extracted.sigma, extracted.sigma.T)

    assert np.all(extracted.event_times[:-1] <= extracted.event_times[1:])


def test_sunab_simple_case():
    n = 600
    cohorts = np.repeat([2010, 2012, 9999], n // 3)
    periods = np.tile([2008, 2009, 2010, 2011, 2012, 2013], n // 6)

    treated = (cohorts != 9999) & (periods >= cohorts)
    y = 1.0 + 0.1 * periods + 0.5 * treated + np.random.normal(0, 0.1, n)

    covariates = np.column_stack([np.ones(n), pd.get_dummies(periods).values[:, 1:]])

    result = sunab(cohort=cohorts, period=periods, outcome=y, covariates=covariates, ref_period=-1)

    assert isinstance(result.att_by_event, np.ndarray)
    assert isinstance(result.vcov, np.ndarray)
    assert len(result.event_times) > 0

    post_treatment_idx = result.event_times >= 0
    assert np.mean(result.att_by_event[post_treatment_idx]) > 0.3


def test_extract_coefficients_patterns():
    patterns_and_names = [
        (
            r"sunab::(\w+)::([-\d]+)",
            ["const", "sunab::2010::-2", "sunab::2010::0", "sunab::2012::-2", "sunab::2012::0"],
        ),
        (
            r"cohort_(\w+)_period_([-\d]+)",
            ["const", "cohort_2010_period_-2", "cohort_2010_period_0", "cohort_2012_period_-2", "cohort_2012_period_0"],
        ),
        (r"g(\d+)_t([-\d]+)", ["const", "g2010_t-2", "g2010_t0", "g2012_t-2", "g2012_t0"]),
    ]

    for pattern, names in patterns_and_names:
        X = np.random.randn(100, len(names))
        y = X @ np.random.randn(len(names)) + np.random.randn(100)

        model = sm.OLS(y, X)
        result = model.fit()
        result._results.params = pd.Series(result.params, index=names)

        coefs = extract_sunab_coefficients(result, pattern=pattern)
        assert len(coefs.beta) == 2
        assert list(coefs.event_times) == [-2, 0]


def test_weighted_sunab_estimation(sunab_panel_data):
    df = sunab_panel_data

    weights = np.random.uniform(0.5, 1.5, len(df))

    covariates = np.column_stack([np.ones(len(df)), pd.get_dummies(df["period"]).values[:, 1:]])

    result = sunab(
        cohort=df["cohort"].values,
        period=df["period"].values,
        outcome=df["y"].values,
        covariates=covariates,
        weights=weights,
        ref_period=-1,
    )

    assert result.att_by_event is not None
    assert result.vcov is not None
    assert result.vcov.shape[0] == len(result.event_times)


def test_no_sunab_coefficients_error():
    X = np.column_stack([np.ones(100), np.random.randn(100)])
    y = X @ np.array([1, 0.5]) + np.random.randn(100)

    model = sm.OLS(y, X)
    result = model.fit()
    result._results.params = pd.Series(result.params, index=["const", "x1"])

    with pytest.raises(ValueError, match="Could not detect Sun & Abraham"):
        extract_sunab_coefficients(result)


def test_binning_functionality():
    n = 800
    cohorts = np.repeat([2010, 2015, 9999], [300, 300, 200])
    periods = np.tile(np.arange(2005, 2021), n // 16)[:n]

    treated = (cohorts != 9999) & (periods >= cohorts)
    y = 0.5 + 0.05 * periods + 0.3 * treated + np.random.normal(0, 0.2, n)

    covariates = np.column_stack([np.ones(n), periods - 2005])

    result = sunab(cohort=cohorts, period=periods, outcome=y, covariates=covariates, ref_period=-1, bin_rel="bin::3")

    assert len(result.event_times) < 15
    assert result.att_by_event is not None


@pytest.mark.parametrize("ref_period", [-1, -2, [-1, -2]])
def test_different_reference_periods(sunab_panel_data, ref_period):
    df = sunab_panel_data

    covariates = np.ones((len(df), 1))

    result = sunab(
        cohort=df["cohort"].values,
        period=df["period"].values,
        outcome=df["y"].values,
        covariates=covariates,
        ref_period=ref_period,
    )

    if isinstance(ref_period, list):
        for rp in ref_period:
            assert rp not in result.event_times
    else:
        assert ref_period not in result.event_times
