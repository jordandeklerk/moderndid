"""Tests for panel treatment effect estimators."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.didcont.estimation.container import AttgtResult
from moderndid.didcont.estimation.estimators import did_attgt, pte_attgt


def test_did_attgt_basic_functionality(simple_panel_data):
    result = did_attgt(simple_panel_data)

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None
    assert len(result.inf_func) == len(simple_panel_data) // 2
    assert result.extra_gt_returns is None


def test_did_attgt_with_covariates(panel_data_with_covariates_estimators):
    result = did_attgt(panel_data_with_covariates_estimators, xformula="~ x1 + x2")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None
    assert np.isfinite(result.attgt)


def test_did_attgt_intercept_only(simple_panel_data):
    result = did_attgt(simple_panel_data, xformula="~1")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_did_attgt_missing_periods():
    df = pl.DataFrame(
        {"id": [1, 2, 3], "D": [1, 0, 1], "period": [0, 0, 0], "name": ["pre", "pre", "pre"], "Y": [1.0, 2.0, 3.0]}
    )

    with pytest.raises(ValueError, match="Data must contain both 'pre' and 'post' periods"):
        did_attgt(df)


def test_pte_attgt_basic_functionality(simple_panel_data):
    result = pte_attgt(simple_panel_data)

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None
    assert result.extra_gt_returns is None


def test_pte_attgt_with_dr_method(panel_data_with_covariates_estimators):
    result = pte_attgt(panel_data_with_covariates_estimators, xformula="~ x1 + x2", est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_with_reg_method(panel_data_with_covariates_estimators):
    result = pte_attgt(panel_data_with_covariates_estimators, xformula="~ x1 + x2", est_method="reg")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_with_lagged_outcome(panel_data_with_covariates_estimators):
    result = pte_attgt(panel_data_with_covariates_estimators, xformula="~ x1", lagged_outcome_cov=True, est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_with_d_outcome(simple_panel_data):
    result = pte_attgt(simple_panel_data, d_outcome=True)

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_with_d_covs_formula(panel_data_with_covariates_estimators):
    result = pte_attgt(panel_data_with_covariates_estimators, xformula="~ x1", d_covs_formula="~ x2", est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_invalid_method():
    df = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "D": [1, 1, 0, 0],
            "period": [0, 1, 0, 1],
            "name": ["pre", "post", "pre", "post"],
            "Y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="Unsupported estimation method"):
        pte_attgt(df, est_method="invalid")


def test_pte_attgt_with_weights():
    np.random.seed(42)
    n_units = 100

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)
    treatment = np.repeat(np.random.binomial(1, 0.5, n_units), 2)
    weights = np.repeat(np.random.uniform(0.5, 1.5, n_units), 2)

    pre_outcome = np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pl.DataFrame(
        {"id": id_vals, "D": treatment, "period": period_vals, "name": name_vals, "Y": y_vals, ".w": weights}
    )

    result = pte_attgt(df)

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_collinearity_handling(panel_data_with_covariates_estimators):
    df = panel_data_with_covariates_estimators.clone()
    df = df.with_columns((pl.col("x1") * 2).alias("x3"))

    result = pte_attgt(df, xformula="~ x1 + x2 + x3", est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_propensity_score_overlap_issue():
    np.random.seed(42)
    n_units = 100

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)

    x_extreme = np.repeat(np.random.normal(0, 5, n_units), 2)
    treatment = np.repeat((x_extreme[:n_units] > 3).astype(int), 2)

    group_vals = np.repeat(2004, n_units * 2)

    pre_outcome = np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pl.DataFrame(
        {
            "id": id_vals,
            "D": treatment,
            "G": group_vals,
            "period": period_vals,
            "name": name_vals,
            "Y": y_vals,
            "x_extreme": x_extreme,
        }
    )

    result = pte_attgt(df, xformula="~ x_extreme", est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)


def test_pte_attgt_all_treated():
    np.random.seed(42)
    df = pl.DataFrame(
        {
            "id": np.repeat([1, 2, 3], 2),
            "D": np.ones(6),
            "period": np.tile([0, 1], 3),
            "name": np.tile(["pre", "post"], 3),
            "Y": np.random.normal(0, 1, 6),
        }
    )

    result = pte_attgt(df, est_method="reg")

    assert isinstance(result, AttgtResult)


def test_pte_attgt_all_control():
    np.random.seed(42)
    df = pl.DataFrame(
        {
            "id": np.repeat([1, 2, 3], 2),
            "D": np.zeros(6),
            "period": np.tile([0, 1], 3),
            "name": np.tile(["pre", "post"], 3),
            "Y": np.random.normal(0, 1, 6),
        }
    )

    result = pte_attgt(df, est_method="reg")

    assert isinstance(result, AttgtResult)


def test_pte_attgt_with_group_info(panel_data_with_groups):
    result = pte_attgt(panel_data_with_groups, est_method="dr")

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None


def test_pte_attgt_combined_formulas():
    np.random.seed(42)
    n_units = 50

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)
    treatment = np.repeat(np.random.binomial(1, 0.5, n_units), 2)

    x1 = np.repeat(np.random.normal(0, 1, n_units), 2)
    x2 = np.repeat(np.random.normal(0, 1, n_units), 2)
    x3 = np.repeat(np.random.normal(0, 1, n_units), 2)

    pre_outcome = x1[:n_units] + np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pl.DataFrame(
        {
            "id": id_vals,
            "D": treatment,
            "period": period_vals,
            "name": name_vals,
            "Y": y_vals,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )

    result1 = pte_attgt(df, xformula="~ x1", d_covs_formula="~ x2")
    assert isinstance(result1, AttgtResult)

    result2 = pte_attgt(df, xformula="~1", d_covs_formula="~ x1 + x2")
    assert isinstance(result2, AttgtResult)

    result3 = pte_attgt(df, xformula="~ x1 + x2", d_covs_formula="~-1")
    assert isinstance(result3, AttgtResult)

    result4 = pte_attgt(df, xformula="~1", d_covs_formula="~-1", lagged_outcome_cov=True)
    assert isinstance(result4, AttgtResult)


def test_empty_data():
    df = pl.DataFrame({"id": [], "D": [], "period": [], "name": [], "Y": []})

    with pytest.raises((ValueError, KeyError)):
        did_attgt(df)

    with pytest.raises((ValueError, KeyError)):
        pte_attgt(df)


def test_single_unit_data():
    df = pl.DataFrame({"id": [1, 1], "D": [1, 1], "period": [0, 1], "name": ["pre", "post"], "Y": [1.0, 2.0]})

    with pytest.raises((ValueError, IndexError)):
        did_attgt(df)


@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_pte_attgt_estimation_methods(panel_data_with_covariates_estimators, est_method):
    result = pte_attgt(panel_data_with_covariates_estimators, xformula="~ x1 + x2", est_method=est_method)

    assert isinstance(result, AttgtResult)
    assert isinstance(result.attgt, float)
    assert result.inf_func is not None
    assert len(result.inf_func) == len(panel_data_with_covariates_estimators) // 2
