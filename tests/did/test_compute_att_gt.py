# pylint: disable=redefined-outer-name
"""Tests for computing ATT-GT."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from moderndid import load_mpdta
from moderndid.did.compute_att_gt import (
    ATTgtResult,
    ComputeATTgtResult,
    compute_att_gt,
    get_did_cohort_index,
    run_att_gt_estimation,
    run_drdid,
)
from moderndid.did.preprocess.models import DIDConfig
from moderndid.did.preprocess_did import preprocess_did


@pytest.fixture
def mpdta_data():
    df = load_mpdta()
    return df


@pytest.fixture
def panel_data(mpdta_data):
    return preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        est_method="dr",
    )


def test_attgt_result_structure():
    result = ATTgtResult(att=1.5, group=2020.0, year=2021.0, post=1)
    assert result.att == 1.5
    assert result.group == 2020.0
    assert result.year == 2021.0
    assert result.post == 1


def test_attgt_result_immutability():
    result = ATTgtResult(att=1.0, group=2020.0, year=2021.0, post=0)
    with pytest.raises(AttributeError):
        result.att = 2.0


def test_compute_attgt_result_structure():
    att_results = [
        ATTgtResult(att=1.0, group=2020.0, year=2021.0, post=1),
        ATTgtResult(att=1.5, group=2021.0, year=2022.0, post=1),
    ]
    inf_funcs = sp.csr_matrix(np.random.randn(100, 2))

    result = ComputeATTgtResult(attgt_list=att_results, influence_functions=inf_funcs)
    assert len(result.attgt_list) == 2
    assert isinstance(result.attgt_list[0], ATTgtResult)
    assert result.influence_functions.shape == (100, 2)
    assert isinstance(result.influence_functions, sp.csr_matrix)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_get_did_cohort_index_panel(mpdta_data, control_group):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group=control_group,
    )

    cohort_index = get_did_cohort_index(
        group_idx=0,
        time_idx=1,
        time_factor=0,
        pre_treatment_idx=0,
        data=data,
    )

    assert isinstance(cohort_index, np.ndarray)
    assert len(cohort_index) == data.config.id_count
    assert set(cohort_index[~np.isnan(cohort_index)]).issubset({0, 1})

    assert np.sum(cohort_index == 1) > 0
    assert np.sum(cohort_index == 0) > 0


def test_get_did_cohort_index_cross_section(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=False,
        control_group="nevertreated",
    )

    cohort_index = get_did_cohort_index(
        group_idx=0,
        time_idx=1,
        time_factor=0,
        pre_treatment_idx=0,
        data=data,
    )

    assert isinstance(cohort_index, np.ndarray)
    assert cohort_index.shape[0] == len(data.data)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_run_drdid_panel_methods(mpdta_data, est_method):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        est_method=est_method,
    )

    n_units = 100
    cohort_data = {
        "D": np.random.choice([0, 1], n_units, p=[0.7, 0.3]),
        "y1": np.random.randn(n_units) + 0.5,
        "y0": np.random.randn(n_units),
        "weights": np.ones(n_units),
    }
    covariates = np.column_stack([np.ones(n_units), np.random.randn(n_units)])

    result = run_drdid(cohort_data, covariates, data)

    assert "att" in result
    assert "inf_func" in result
    assert isinstance(result["att"], float | np.floating)
    assert isinstance(result["inf_func"], np.ndarray)
    assert len(result["inf_func"]) == n_units


def test_run_drdid_with_missing_data(panel_data):
    n_units = 100
    cohort_data = {
        "D": np.array([0, 1, np.nan] * 33 + [1]),
        "y1": np.random.randn(n_units),
        "y0": np.random.randn(n_units),
        "weights": np.ones(n_units),
    }
    covariates = np.column_stack([np.ones(n_units), np.random.randn(n_units)])

    result = run_drdid(cohort_data, covariates, panel_data)

    assert "att" in result
    assert "inf_func" in result
    assert len(result["inf_func"]) == n_units


def test_run_drdid_all_missing(panel_data):
    n_units = 50
    cohort_data = {
        "D": np.full(n_units, np.nan),
        "y1": np.random.randn(n_units),
        "y0": np.random.randn(n_units),
        "weights": np.ones(n_units),
    }
    covariates = np.ones((n_units, 1))

    result = run_drdid(cohort_data, covariates, panel_data)

    assert "att" in result
    assert "inf_func" in result
    assert np.isnan(result["att"])
    assert np.all(result["inf_func"] == 0)


@pytest.mark.parametrize("panel", [True, False])
def test_run_drdid_panel_vs_cross_section(mpdta_data, panel):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=panel,
        control_group="nevertreated",
        est_method="dr",
    )

    n_units = 100
    if panel:
        cohort_data = {
            "D": np.random.choice([0, 1], n_units),
            "y1": np.random.randn(n_units),
            "y0": np.random.randn(n_units),
            "weights": np.ones(n_units),
        }
    else:
        cohort_data = {
            "D": np.random.choice([0, 1], n_units),
            "y": np.random.randn(n_units),
            "post": np.random.choice([0, 1], n_units),
            "weights": np.ones(n_units),
        }

    covariates = np.column_stack([np.ones(n_units), np.random.randn(n_units)])
    result = run_drdid(cohort_data, covariates, data)

    assert "att" in result
    assert "inf_func" in result


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_run_att_gt_estimation_base_periods(mpdta_data, base_period):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period=base_period,
    )

    time_idx = 1 if base_period == "varying" else 2
    result = run_att_gt_estimation(group_idx=0, time_idx=time_idx, data=data)

    if result is not None:
        assert "att" in result
        assert "inf_func" in result
        assert isinstance(result["att"], float | np.floating)
        assert isinstance(result["inf_func"], np.ndarray)


def test_run_att_gt_estimation_no_control_units(mpdta_data):
    df_all_treated = mpdta_data.copy()
    df_all_treated["first.treat"] = 2004

    with pytest.raises(ValueError, match="No valid groups"):
        preprocess_did(
            df_all_treated,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            xformla="~lpop",
            panel=True,
            control_group="nevertreated",
        )


def test_compute_att_gt_full_run(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert isinstance(result.attgt_list, list)
    assert len(result.attgt_list) > 0

    for att_result in result.attgt_list:
        assert isinstance(att_result, ATTgtResult)
        assert isinstance(att_result.att, float | np.floating)
        assert isinstance(att_result.group, float | np.floating)
        assert isinstance(att_result.year, int | float | np.integer | np.floating)
        assert att_result.post in [0, 1]

    assert isinstance(result.influence_functions, sp.csr_matrix)
    assert result.influence_functions.shape[0] == data.config.id_count
    assert result.influence_functions.shape[1] == len(result.attgt_list)


def test_compute_att_gt_universal_base(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="universal",
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)

    pre_treatment_atts = [r for r in result.attgt_list if r.post == 0]
    assert len(pre_treatment_atts) > 0


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_different_methods(mpdta_data, est_method):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        est_method=est_method,
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_compute_att_gt_control_groups(mpdta_data, control_group):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group=control_group,
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


def test_compute_att_gt_cross_section(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=False,
        control_group="nevertreated",
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


def test_compute_att_gt_with_anticipation(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        anticipation=1,
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


def test_compute_att_gt_consistency(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    result1 = compute_att_gt(data)
    result2 = compute_att_gt(data)

    assert len(result1.attgt_list) == len(result2.attgt_list)
    for r1, r2 in zip(result1.attgt_list, result2.attgt_list):
        assert np.isclose(r1.att, r2.att, rtol=1e-10)
        assert r1.group == r2.group
        assert r1.year == r2.year
        assert r1.post == r2.post


def test_compute_att_gt_influence_functions_properties(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    result = compute_att_gt(data)

    inf_funcs_dense = result.influence_functions.toarray()
    for col in range(inf_funcs_dense.shape[1]):
        col_mean = np.mean(inf_funcs_dense[:, col])
        assert np.abs(col_mean) < 0.1


def test_compute_att_gt_sparse_efficiency(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    result = compute_att_gt(data)

    density = result.influence_functions.nnz / (
        result.influence_functions.shape[0] * result.influence_functions.shape[1]
    )
    assert density < 1.0


@pytest.mark.parametrize("anticipation", [0, 1, 2])
def test_compute_att_gt_anticipation_levels(mpdta_data, anticipation):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        anticipation=anticipation,
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


def test_compute_att_gt_no_covariates(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        panel=True,
        control_group="nevertreated",
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert len(result.attgt_list) > 0


def test_compute_att_gt_edge_cases(mpdta_data):
    df_small = mpdta_data.iloc[:100].copy()

    data = preprocess_did(
        df_small,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    result = compute_att_gt(data)

    assert isinstance(result, ComputeATTgtResult)
    assert isinstance(result.attgt_list, list)
    assert isinstance(result.influence_functions, sp.csr_matrix)


def test_cohort_index_edge_cases(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    n_groups = data.config.treated_groups_count
    n_times = len(data.config.time_periods)

    cohort_index = get_did_cohort_index(
        group_idx=n_groups - 1,
        time_idx=n_times - 2,
        time_factor=0,
        pre_treatment_idx=0,
        data=data,
    )
    assert isinstance(cohort_index, np.ndarray)


def test_influence_function_aggregation(mpdta_data):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=False,
        allow_unbalanced_panel=True,
        control_group="nevertreated",
    )

    n_units = 100
    cohort_data = {
        "D": np.random.choice([0, 1], n_units),
        "y": np.random.randn(n_units),
        "post": np.random.choice([0, 1], n_units),
        "weights": np.ones(n_units),
        "rowid": np.repeat(np.arange(50), 2),
    }
    covariates = np.column_stack([np.ones(n_units), np.random.randn(n_units)])

    result = run_drdid(cohort_data, covariates, data)

    assert "att" in result
    assert "inf_func" in result
    assert len(result["inf_func"]) <= n_units


def test_no_variation_in_treatment_timing():
    n = 100
    n_periods = 4

    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n), n_periods),
            "year": np.tile(np.arange(n_periods), n),
            "lemp": np.random.randn(n * n_periods),
            "first.treat": np.repeat(2, n * n_periods),
            "lpop": np.random.randn(n * n_periods),
        }
    )

    config = DIDConfig(
        yname="lemp",
        tname="year",
        idname="id",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
    )

    with pytest.raises(ValueError, match="No valid groups"):
        preprocess_did(
            df,
            yname=config.yname,
            tname=config.tname,
            idname=config.idname,
            gname=config.gname,
            xformla=config.xformla,
            panel=config.panel,
            control_group=config.control_group,
        )
