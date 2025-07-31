# pylint: disable=redefined-outer-name, protected-access
"""Test the medicaid example from the R package HonestDiD."""

import numpy as np
import pandas as pd
import pytest

from tests.helpers import importorskip

pf = importorskip("pyfixest")

from doublediff.didhonest import (
    construct_original_cs,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)


@pytest.fixture(scope="module")
def medicaid_data():
    url = "https://raw.githubusercontent.com/Mixtape-Sessions/Advanced-DID/main/Exercises/Data/ehec_data.dta"
    df = pd.read_stata(url)

    if df["year"].dtype.name == "category":
        df["year"] = df["year"].astype(int)
    if "yexp2" in df.columns and df["yexp2"].dtype.name == "category":
        df["yexp2"] = df["yexp2"].astype(float)
    if df["stfips"].dtype.name == "category":
        df["stfips"] = df["stfips"].astype(str)

    df_nonstaggered = df[(df["year"] < 2016) & (df["yexp2"].isna() | (df["yexp2"] != 2015))].copy()

    df_nonstaggered["D"] = np.where(df_nonstaggered["yexp2"] == 2014, 1, 0)

    years = sorted(df_nonstaggered["year"].unique())
    for year in years:
        if year != 2013:
            df_nonstaggered[f"D_year_{year}"] = df_nonstaggered["D"] * (df_nonstaggered["year"] == year)

    return df_nonstaggered


@pytest.fixture
def event_study_results(medicaid_data):
    years = sorted(medicaid_data["year"].unique())
    interaction_terms = [f"D_year_{year}" for year in years if year != 2013]
    formula = f"dins ~ {' + '.join(interaction_terms)} | stfips + year"

    twfe_results = pf.feols(formula, data=medicaid_data, vcov={"CRV1": "stfips"})

    pre_years = [2008, 2009, 2010, 2011, 2012]
    post_years = [2014, 2015]

    coef_names = [f"D_year_{year}" for year in pre_years + post_years]
    betahat = np.array([twfe_results.coef()[name] for name in coef_names])

    sigma_full = twfe_results._vcov

    all_coef_names = list(twfe_results.coef().index)
    coef_indices = [all_coef_names.index(name) for name in coef_names]

    sigma = sigma_full[np.ix_(coef_indices, coef_indices)]

    return betahat, sigma


def test_event_study_coefficients(event_study_results):
    betahat, _ = event_study_results

    pre_coefs = betahat[:5]
    assert all(np.abs(pre_coefs) < 0.1), "Pre-treatment effects should be small"

    post_coefs = betahat[5:]
    assert all(post_coefs > 0), "Post-treatment effects should be positive"

    assert np.abs(betahat[5] - 0.046) < 0.01, "2014 effect should be around 0.046"
    assert np.abs(betahat[6] - 0.069) < 0.01, "2015 effect should be around 0.069"


def test_relative_magnitudes_sensitivity(event_study_results):
    betahat, sigma = event_study_results

    m_bar_vec_rm = np.arange(0.5, 2.5, 0.5)
    delta_rm_results = create_sensitivity_results_rm(
        betahat=betahat, sigma=sigma, num_pre_periods=5, num_post_periods=2, m_bar_vec=m_bar_vec_rm
    )

    expected_rm = pd.DataFrame(
        {
            "Mbar": [0.5, 1.0, 1.5, 2.0],
            "lb": [0.0241, 0.0171, 0.00859, -0.00107],
            "ub": [0.0673, 0.0720, 0.0796, 0.0883],
        }
    )

    for _, row in expected_rm.iterrows():
        actual = delta_rm_results[delta_rm_results["Mbar"] == row["Mbar"]].iloc[0]
        np.testing.assert_allclose(actual["lb"], row["lb"], atol=1e-3, rtol=0)
        np.testing.assert_allclose(actual["ub"], row["ub"], atol=1e-3, rtol=0)

    breakdown_rows = delta_rm_results[delta_rm_results["lb"] <= 0]
    assert len(breakdown_rows) > 0, "Should find a breakdown value"
    breakdown_mbar = breakdown_rows.iloc[0]["Mbar"]
    assert 1.5 < breakdown_mbar <= 2.0, "Breakdown value should be around 2"


def test_smoothness_sensitivity(event_study_results):
    betahat, sigma = event_study_results

    m_vec_sd = np.arange(0, 0.06, 0.01)
    delta_sd_results = create_sensitivity_results_sm(
        betahat=betahat, sigma=sigma, num_pre_periods=5, num_post_periods=2, m_vec=m_vec_sd
    )

    m0_result = delta_sd_results[delta_sd_results["m"] == 0.0].iloc[0]
    assert m0_result["lb"] > 0, "Lower bound at M=0 should be positive"
    assert m0_result["ub"] > m0_result["lb"], "Upper bound should exceed lower bound"

    for i in range(1, len(delta_sd_results)):
        prev = delta_sd_results.iloc[i - 1]
        curr = delta_sd_results.iloc[i]
        assert curr["lb"] <= prev["lb"] + 1e-6, f"Lower bound should decrease as M increases (M={curr['m']})"
        assert curr["ub"] >= prev["ub"] - 1e-6, f"Upper bound should increase as M increases (M={curr['m']})"

    breakdown_rows = delta_sd_results[delta_sd_results["lb"] <= 0]
    assert len(breakdown_rows) > 0, "Should find a breakdown value"
    breakdown_m = breakdown_rows.iloc[0]["m"]
    assert 0.01 < breakdown_m <= 0.04, f"Breakdown value {breakdown_m} outside expected range"

    assert all(delta_sd_results["method"] == "FLCI"), "All results should use FLCI method"
    assert all(delta_sd_results["delta"] == "DeltaSD"), "All results should use DeltaSD"


def test_original_confidence_interval(event_study_results):
    betahat, sigma = event_study_results

    original_results = construct_original_cs(betahat=betahat, sigma=sigma, num_pre_periods=5, num_post_periods=2)

    assert original_results.method == "Original"
    assert original_results.delta is None
    assert original_results.lb < original_results.ub
    assert 0.02 < original_results.lb < 0.03
    assert 0.05 < original_results.ub < 0.07


def test_average_effect_sensitivity(event_study_results):
    betahat, sigma = event_study_results

    l_vec = np.array([0.5, 0.5])
    delta_rm_results_avg = create_sensitivity_results_rm(
        betahat=betahat,
        sigma=sigma,
        num_pre_periods=5,
        num_post_periods=2,
        m_bar_vec=np.arange(0, 2.5, 0.5),
        l_vec=l_vec,
    )

    assert isinstance(delta_rm_results_avg, pd.DataFrame)
    assert len(delta_rm_results_avg) > 0
    assert "Mbar" in delta_rm_results_avg.columns

    low_mbar_results = delta_rm_results_avg[delta_rm_results_avg["Mbar"] <= 1.0]
    assert all(low_mbar_results["lb"] > 0), "Average effect should be significantly positive for low Mbar"

    original_results_avg = construct_original_cs(
        betahat=betahat, sigma=sigma, num_pre_periods=5, num_post_periods=2, l_vec=l_vec
    )

    assert original_results_avg.lb > 0, "Average effect should be significantly positive"


@pytest.mark.parametrize("method", ["FLCI", "C-LF", "Conditional"])
def test_different_methods(event_study_results, method):
    betahat, sigma = event_study_results

    delta_results = create_sensitivity_results_sm(
        betahat=betahat, sigma=sigma, num_pre_periods=5, num_post_periods=2, m_vec=np.array([0.0, 0.02]), method=method
    )

    assert isinstance(delta_results, pd.DataFrame)
    assert len(delta_results) == 2
    assert all(delta_results["method"] == method)
    assert all(delta_results["lb"] <= delta_results["ub"])
