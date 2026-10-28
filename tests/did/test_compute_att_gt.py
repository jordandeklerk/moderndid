"""Tests for computing ATT-GT."""

import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
import scipy.sparse as sp

from moderndid import (
    ATTgtResult,
    compute_att_gt,
    drdid_panel,
    drdid_rc,
    preprocess_did,
    reg_did_panel,
    reg_did_rc,
    std_ipw_did_panel,
    std_ipw_did_rc,
)
from moderndid.did.compute_att_gt import cell_spec, get_did_cohort_index, run_att_gt_estimation, run_drdid


def test_attgt_result_immutability():
    result = ATTgtResult(att=1.0, group=2020.0, year=2021.0, post=0)

    with pytest.raises(AttributeError, match="can't set attribute"):
        result.att = 2.0


@pytest.mark.parametrize("control_group, controls", [("nevertreated", [np.inf]), ("notyettreated", [np.inf, 2007])])
def test_get_did_cohort_index_panel_marks_treated_and_comparison_units(mpdta_data, control_group, controls):
    data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", control_group=control_group
    )
    cohorts = data.time_invariant_data["first.treat"].to_numpy()
    expected = np.where(cohorts == 2004, 1.0, np.where(np.isin(cohorts, controls), 0.0, np.nan))

    cohort_index = get_did_cohort_index(group_idx=0, time_idx=2, time_factor=1, pre_treatment_idx=0, data=data)

    np.testing.assert_array_equal(cohort_index, expected)


@pytest.mark.parametrize("control_group, controls", [("nevertreated", [np.inf]), ("notyettreated", [np.inf, 2007])])
def test_get_did_cohort_index_cross_section_marks_treated_and_comparison_rows(mpdta_data, control_group, controls):
    data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", gname="first.treat", panel=False, control_group=control_group
    )
    cohorts = data.data["first.treat"].to_numpy()
    in_cell = data.data["year"].is_in([2003, 2006]).to_numpy()
    expected = np.where(in_cell & (cohorts == 2004), 1.0, np.where(in_cell & np.isin(cohorts, controls), 0.0, np.nan))

    cohort_index = get_did_cohort_index(group_idx=0, time_idx=2, time_factor=1, pre_treatment_idx=0, data=data)

    np.testing.assert_array_equal(cohort_index, expected)


def test_cell_spec_leaves_the_treated_cohort_out_of_its_comparison_cohorts(mpdta_data):
    data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", control_group="notyettreated"
    )

    spec = cell_spec(group_idx=2, time_idx=0, config=data.config, cohort_counts=data.cohort_counts)

    np.testing.assert_array_equal(spec.control_cohorts, [2006.0, np.inf])


@pytest.mark.parametrize(
    "est_method, estimator", [("dr", drdid_panel), ("ipw", std_ipw_did_panel), ("reg", reg_did_panel)]
)
def test_run_drdid_panel_methods(mpdta_data, est_method, estimator):
    data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", est_method=est_method
    )
    rng = np.random.default_rng(3)
    x = rng.normal(size=120)
    treated = rng.random(120) < 1 / (1 + np.exp(-x))
    valid = np.arange(120) % 8 != 0
    y0 = x + rng.normal(size=120)
    cohort_data = {
        "D": np.where(valid, treated, np.nan),
        "y1": y0 + x + treated + rng.normal(size=120),
        "y0": y0,
        "weights": rng.uniform(0.5, 2.0, 120),
    }
    covariates = np.column_stack([np.ones(120), x])
    direct = estimator(
        y1=cohort_data["y1"][valid],
        y0=y0[valid],
        d=treated[valid].astype(float),
        covariates=covariates[valid],
        i_weights=cohort_data["weights"][valid],
        influence_func=True,
    )

    result = run_drdid(cohort_data, covariates, data)

    np.testing.assert_allclose(result["att"], direct.att, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result["inf_func"][valid], 120 / 105 * direct.att_inf_func, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(result["inf_func"][~valid], 0.0)


def test_run_drdid_with_missing_data(panel_data, four_means):
    rng = np.random.default_rng(5)
    d = np.array([0, 1, np.nan] * 33 + [1])
    y0 = rng.normal(size=100)
    weights = rng.uniform(0.5, 2.0, 100)
    cohort_data = {"D": d, "y1": y0 + 0.5 + rng.normal(size=100), "y0": y0, "weights": weights}
    later = np.repeat([True, False], 100)
    att, share = four_means(
        np.r_[cohort_data["y1"], y0], np.tile(d == 1, 2), np.tile(d == 0, 2), later, ~later, np.tile(weights, 2)
    )

    result = run_drdid(cohort_data, np.ones((100, 1)), panel_data)

    np.testing.assert_allclose(result["att"], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["inf_func"], 100 * (share[:100] + share[100:]), rtol=1e-10, atol=0)


def test_run_drdid_all_missing(panel_data):
    rng = np.random.default_rng(7)
    cohort_data = {
        "D": np.full(50, np.nan),
        "y1": rng.normal(size=50),
        "y0": rng.normal(size=50),
        "weights": np.ones(50),
    }

    result = run_drdid(cohort_data, np.ones((50, 1)), panel_data)

    assert np.isnan(result["att"])
    np.testing.assert_array_equal(result["inf_func"], np.full(50, np.nan))


def test_run_drdid_cross_section_with_missing_data(mpdta_data, four_means):
    data = preprocess_did(mpdta_data, yname="lemp", tname="year", gname="first.treat", panel=False)
    rng = np.random.default_rng(9)
    row = np.arange(120)
    d = np.where(row % 6 == 5, np.nan, (row % 3 == 0).astype(float))
    post = row % 2
    weights = rng.uniform(0.5, 2.0, 120)
    cohort_data = {"D": d, "y": rng.normal(size=120) + post + (row % 3 == 0) * post, "post": post, "weights": weights}
    att, share = four_means(cohort_data["y"], d == 1, d == 0, post == 1, post == 0, weights)

    result = run_drdid(cohort_data, np.ones((120, 1)), data)

    np.testing.assert_allclose(result["att"], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["inf_func"], 120 * share, rtol=1e-10, atol=0)


def test_run_drdid_unbalanced_panel_sums_each_unit_rows(mpdta_unbalanced, four_means):
    data = preprocess_did(
        mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
    )
    rng = np.random.default_rng(0)
    unit = np.repeat(np.arange(60), 2)
    post = np.tile([0, 1], 60)
    observed = (unit % 7 != 6) | (post == 0)
    unit = unit[observed]
    post = post[observed]
    treated = unit % 3 == 0
    d = np.where(unit % 5 == 4, np.nan, treated.astype(float))
    weights = rng.uniform(0.5, 2.0, unit.size)
    cohort_data = {
        "D": d,
        "y": rng.normal(size=60)[unit] + post + treated * post + rng.normal(size=unit.size),
        "post": post,
        "weights": weights,
        "unit_index": unit,
    }
    att, share = four_means(cohort_data["y"], d == 1, d == 0, post == 1, post == 0, weights)
    expected = 500 * np.bincount(unit, weights=share, minlength=500)

    result = run_drdid(cohort_data, np.ones((unit.size, 1)), data)

    np.testing.assert_allclose(result["att"], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["inf_func"], expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("base_period, time_idx, base", [("varying", 0, 2003), ("universal", 1, 2005)])
def test_run_att_gt_estimation_pre_treatment_cell_follows_base_period(
    mpdta_pop_weighted, cells_from_weighted_means, base_period, time_idx, base
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        base_period=base_period,
    )
    att, inf = cells_from_weighted_means(
        mpdta_pop_weighted, data.time_invariant_data.select("countyreal"), [(2006, 2004, base, [0])]
    )

    result = run_att_gt_estimation(group_idx=1, time_idx=time_idx, data=data)

    np.testing.assert_allclose(result["att"], att[0], rtol=1e-10, atol=0)
    np.testing.assert_allclose(result["inf_func"], inf[:, 0], rtol=1e-10, atol=0)


@pytest.mark.filterwarnings("ignore:No never-treated group:UserWarning")
def test_preprocess_did_single_cohort_without_never_treated_units_has_no_groups(mpdta_data):
    data = mpdta_data.with_columns(pl.lit(2004).alias("first.treat"))

    with pytest.raises(ValueError, match="No valid groups"):
        preprocess_did(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_compute_att_gt_matches_weighted_mean_changes_with_each_control_group(
    mpdta_pop_weighted, mpdta_cells, cells_from_weighted_means, control_group
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        control_group=control_group,
    )
    cells = mpdta_cells(control_group=control_group)
    att, inf = cells_from_weighted_means(mpdta_pop_weighted, data.time_invariant_data.select("countyreal"), cells)

    result = compute_att_gt(data)

    assert [(cell.group, cell.year, cell.post) for cell in result.attgt_list] == [
        (group, period, int(period >= group)) for group, period, _, _ in cells
    ]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    assert isinstance(result.influence_functions, sp.csr_matrix)
    assert result.influence_functions.nnz == np.count_nonzero(inf)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_compute_att_gt_universal_base_compares_every_period_with_the_last_untreated_period(
    mpdta_pop_weighted, mpdta_cells, cells_from_weighted_means, control_group
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        control_group=control_group,
        base_period="universal",
    )
    cells = mpdta_cells(base_period="universal", control_group=control_group)
    att, inf = cells_from_weighted_means(mpdta_pop_weighted, data.time_invariant_data.select("countyreal"), cells)

    result = compute_att_gt(data)

    assert [(cell.group, cell.year, cell.post) for cell in result.attgt_list] == [
        (group, period, int(period >= group)) for group, period, _, _ in cells
    ]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    assert result.influence_functions.nnz == np.count_nonzero(inf)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.filterwarnings("ignore:Dropped 20 units that were already treated:UserWarning")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("anticipation", [1, 2])
def test_compute_att_gt_anticipation_moves_base_periods_and_comparison_groups(
    mpdta_pop_weighted, mpdta_cells, cells_from_weighted_means, anticipation, control_group
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        control_group=control_group,
        anticipation=anticipation,
    )
    rows = mpdta_pop_weighted.filter(pl.col("first.treat") != 2004)
    cells = mpdta_cells(control_group=control_group, anticipation=anticipation)
    att, inf = cells_from_weighted_means(rows, data.time_invariant_data.select("countyreal"), cells)

    result = compute_att_gt(data)

    assert [(cell.group, cell.year, cell.post) for cell in result.attgt_list] == [
        (group, period, int(period >= group)) for group, period, _, _ in cells
    ]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.filterwarnings("ignore:Dropped 20 units that were already treated:UserWarning")
@pytest.mark.parametrize(
    "base_period, control_group, anticipation",
    [("varying", "nevertreated", 0), ("varying", "notyettreated", 1), ("universal", "notyettreated", 0)],
)
def test_compute_att_gt_cross_section_matches_weighted_mean_outcomes(
    mpdta_pop_weighted, mpdta_cells, cells_from_weighted_means, base_period, control_group, anticipation
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        panel=False,
        control_group=control_group,
        anticipation=anticipation,
        base_period=base_period,
    )
    rows = mpdta_pop_weighted.filter((pl.col("first.treat") == 0) | (pl.col("first.treat") > 2003 + anticipation))
    cells = mpdta_cells(base_period=base_period, control_group=control_group, anticipation=anticipation)
    att, inf = cells_from_weighted_means(rows, data.data.select("countyreal", "year"), cells)

    result = compute_att_gt(data)

    assert [(cell.group, cell.year, cell.post) for cell in result.attgt_list] == [
        (group, period, int(period >= group)) for group, period, _, _ in cells
    ]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


def test_compute_att_gt_cross_section_warns_when_a_cell_period_lacks_treated_or_comparison_rows(
    rcs_with_empty_cell_periods, four_means
):
    rows = rcs_with_empty_cell_periods
    data = preprocess_did(rows, yname="y", tname="t", gname="g", panel=False)
    period = rows["t"].to_numpy()
    att, _ = four_means(
        rows["y"].to_numpy(),
        (rows["g"] == 5).to_numpy(),
        (rows["g"] == 0).to_numpy(),
        period == 3,
        period == 2,
        np.ones(rows.height),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == [
        "No treated units in group 3 in time period 2",
        "No control units for group 3 in pre-treatment period",
        "No treated units in group 3 in pre-treatment period",
        "No treated units in group 3 in pre-treatment period",
        "No control units for group 3 in time period 4",
        "No treated units in group 3 in pre-treatment period",
        "No control units for group 5 in pre-treatment period",
        "No control units for group 5 in time period 4",
        "No control units for group 5 in pre-treatment period",
    ]
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(5, 3)]
    np.testing.assert_allclose(result.attgt_list[0].att, att, rtol=1e-10, atol=0)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize(
    "est_method, panel, estimator",
    [
        ("dr", True, drdid_panel),
        ("ipw", True, std_ipw_did_panel),
        ("reg", True, reg_did_panel),
        ("dr", False, drdid_rc),
        ("ipw", False, std_ipw_did_rc),
        ("reg", False, reg_did_rc),
        (drdid_panel, True, drdid_panel),
        (drdid_rc, False, drdid_rc),
    ],
)
def test_compute_att_gt_runs_the_chosen_estimator_in_every_cell(
    mpdta_pop_weighted, mpdta_cells, cells_from_estimator, est_method, panel, estimator
):
    data = preprocess_did(
        mpdta_pop_weighted,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        weightsname="pop",
        est_method=est_method,
        panel=panel,
    )
    keys = data.time_invariant_data.select("countyreal") if panel else data.data.select("countyreal", "year")
    att, inf = cells_from_estimator(mpdta_pop_weighted, keys, mpdta_cells(), estimator, panel=panel)

    result = compute_att_gt(data)

    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_compute_att_gt_takes_panel_covariates_from_the_earlier_period(
    mpdta_pop_weighted, mpdta_cells, cells_from_estimator, base_period
):
    rows = mpdta_pop_weighted.with_columns(pl.col("lpop") + ((pl.col("countyreal") + pl.col("year")) % 3) / 2)
    data = preprocess_did(
        rows,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        weightsname="pop",
        est_method="reg",
        base_period=base_period,
    )
    att, inf = cells_from_estimator(
        rows, data.time_invariant_data.select("countyreal"), mpdta_cells(base_period=base_period), reg_did_panel
    )

    result = compute_att_gt(data)

    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


def test_compute_att_gt_influence_functions_match_weight_derivative(
    mpdta_pop_weighted, mpdta_atts_with_scaled_unit_weights, weight_derivative
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
    }
    data = preprocess_did(mpdta_pop_weighted, **spec)
    units = data.time_invariant_data["countyreal"]
    cohorts = data.time_invariant_data["first.treat"].to_numpy()
    edges = np.flatnonzero(cohorts[1:] != cohorts[:-1])
    rows = np.r_[0, edges, edges + 1, len(units) - 1]

    result = compute_att_gt(data)
    expected = weight_derivative(
        lambda factors: mpdta_atts_with_scaled_unit_weights(units, factors, **spec), len(units), rows
    )

    np.testing.assert_allclose(result.influence_functions.toarray()[rows], expected, rtol=1e-6, atol=1e-8)


@pytest.mark.filterwarnings("ignore:No never-treated group:UserWarning")
def test_compute_att_gt_without_never_treated_units_compares_with_the_latest_cohort(
    mpdta_pop_weighted, cells_from_weighted_means
):
    rows = mpdta_pop_weighted.head(100)
    data = preprocess_did(rows, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="pop")
    cells = [(2006, 2004, 2003, [2007]), (2006, 2005, 2004, [2007]), (2006, 2006, 2005, [2007])]
    att, inf = cells_from_weighted_means(rows, data.time_invariant_data.select("countyreal"), cells)

    result = compute_att_gt(data)

    assert [(cell.group, cell.year, cell.post) for cell in result.attgt_list] == [
        (2006, 2004, 0),
        (2006, 2005, 0),
        (2006, 2006, 1),
    ]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
def test_compute_att_gt_cross_section_keys_every_row_whatever_idname(mpdta_data):
    spec = dict(yname="lemp", tname="year", gname="first.treat", xformla="~lpop", panel=False)

    plain = compute_att_gt(preprocess_did(mpdta_data, **spec))
    with_id = compute_att_gt(preprocess_did(mpdta_data, idname="countyreal", **spec))

    assert with_id.influence_functions.shape == (mpdta_data.height, len(with_id.attgt_list))
    np.testing.assert_array_equal(with_id.influence_functions.toarray(), plain.influence_functions.toarray())


def test_compute_att_gt_unbalanced_rows_follow_time_invariant_data(
    mpdta_unbalanced, mpdta_cells, cells_from_weighted_means
):
    rows = mpdta_unbalanced.with_columns(pl.col("lpop").exp().alias("pop"))
    data = preprocess_did(
        rows,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="pop",
        allow_unbalanced_panel=True,
    )
    cells = mpdta_cells()
    att, inf = cells_from_weighted_means(rows, data.time_invariant_data.select("countyreal"), cells)

    result = compute_att_gt(data)

    assert not data.config.panel
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(group, period) for group, period, _, _ in cells]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


def test_run_att_gt_estimation_unbalanced_matches_compute_att_gt(mpdta_unbalanced):
    data = preprocess_did(
        mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
        control_group="nevertreated",
    )

    full = compute_att_gt(data)
    cell = run_att_gt_estimation(group_idx=0, time_idx=1, data=data)

    assert (full.attgt_list[1].group, full.attgt_list[1].year) == (2004, 2005)
    assert cell["att"] == full.attgt_list[1].att
    np.testing.assert_array_equal(cell["inf_func"], full.influence_functions.toarray()[:, 1])


def test_compute_att_gt_skips_each_cell_whose_treated_regression_is_unidentified_with_one_warning(
    rcs_small_cohort_period,
):
    data = preprocess_did(
        rcs_small_cohort_period,
        yname="y",
        tname="t",
        idname="id",
        gname="g",
        xformla="~ x0 + x1 + x2 + x3 + x4",
        panel=False,
        est_method="dr",
    )
    advice = (
        "Pass fewer covariates. The estimators drdid_imp_rc and drdid_trad_rc fit no regression for the treated "
        "group and accept this design."
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert result.attgt_list == []
    assert result.influence_functions.shape == (rcs_small_cohort_period.height, 0)
    assert [str(warning.message) for warning in caught] == [
        "The 2x2 comparison for group 3 in time period 2 failed. The treated group in the post-period has fewer "
        f"observations (4) than regression coefficients (6). {advice}",
        "The 2x2 comparison for group 3 in time period 3 failed. The treated group in the pre-period has fewer "
        f"observations (4) than regression coefficients (6). {advice}",
    ]


def test_compute_att_gt_overlap_screen_ignores_units_without_weight(zero_weight_outlier_panel):
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "xformla": "~x", "weightsname": "w"}
    data = preprocess_did(zero_weight_outlier_panel, **spec)
    without_outlier = compute_att_gt(preprocess_did(zero_weight_outlier_panel.filter(pl.col("w") > 0), **spec))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(3, 2), (3, 3)]
    np.testing.assert_allclose(
        [cell.att for cell in result.attgt_list], [cell.att for cell in without_outlier.attgt_list], rtol=1e-10, atol=0
    )


def test_compute_att_gt_stays_silent_about_weights_that_differ_by_rounding(mpdta_pop_weighted):
    rows = mpdta_pop_weighted.with_columns(pl.col("pop") * (1 + 1e-12 * (pl.col("year") - 2005)))
    data = preprocess_did(rows, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="pop")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_att_gt(data)

    assert rows.group_by("countyreal").agg(pl.col("pop").n_unique())["pop"].min() == 5
    assert [str(warning.message) for warning in caught] == []


@pytest.mark.parametrize(
    "base_period,control_group",
    [("varying", "nevertreated"), ("universal", "nevertreated"), ("varying", "notyettreated")],
)
def test_compute_att_gt_parallel_matches_sequential(mpdta_data, base_period, control_group):
    data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group=control_group,
        base_period=base_period,
    )

    result_seq = compute_att_gt(data, n_jobs=1)
    result_par = compute_att_gt(data, n_jobs=2)

    assert result_seq.attgt_list == result_par.attgt_list
    np.testing.assert_array_equal(result_seq.influence_functions.toarray(), result_par.influence_functions.toarray())


@pytest.mark.benchmark
def test_benchmark_compute_att_gt_sequential(
    benchmark, _preprocessed_panel, mpdta_data, mpdta_cells, cells_from_estimator
):
    keys = _preprocessed_panel.time_invariant_data.select("countyreal")
    att, inf = cells_from_estimator(mpdta_data, keys, mpdta_cells(), drdid_panel, weights=None)

    result = benchmark.pedantic(
        compute_att_gt, args=(_preprocessed_panel,), kwargs={"n_jobs": 1}, rounds=3, warmup_rounds=1
    )

    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.benchmark
def test_benchmark_compute_att_gt_parallel(
    benchmark, _preprocessed_panel, mpdta_data, mpdta_cells, cells_from_estimator
):
    keys = _preprocessed_panel.time_invariant_data.select("countyreal")
    att, inf = cells_from_estimator(mpdta_data, keys, mpdta_cells(), drdid_panel, weights=None)

    result = benchmark.pedantic(
        compute_att_gt, args=(_preprocessed_panel,), kwargs={"n_jobs": -1}, rounds=3, warmup_rounds=1
    )

    np.testing.assert_allclose([cell.att for cell in result.attgt_list], att, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result.influence_functions.toarray(), inf, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("data_fixture", ["wide_panel", "wider_panel"])
def test_compute_att_gt_wide_panel_matches_least_squares_and_newton_logit_fits(
    request, reference_cell_att, data_fixture, est_method
):
    rows = request.getfixturevalue(data_fixture)
    covariates = [column for column in rows.columns if column.startswith("cov")]
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method=est_method,
    )
    cells = [(2, 2, 1, [0]), (2, 3, 1, [0]), (3, 2, 1, [0]), (3, 3, 2, [0])]
    expected = [reference_cell_att(rows, covariates, cell, est_method) for cell in cells]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_wide_cross_section_matches_least_squares_and_newton_logit_fits(
    wide_cross_section, reference_cell_att, est_method
):
    covariates = [column for column in wide_cross_section.columns if column.startswith("cov")]
    data = preprocess_did(
        wide_cross_section,
        yname="y",
        tname="time",
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method=est_method,
        panel=False,
    )
    cells = [(2, 2, 1, [0]), (2, 3, 1, [0]), (3, 2, 1, [0]), (3, 3, 2, [0])]
    expected = [reference_cell_att(wide_cross_section, covariates, cell, est_method, panel=False) for cell in cells]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_wide_panel_influence_functions_match_weight_derivative(
    wide_panel, wide_panel_atts_with_scaled_unit_weights, weight_derivative, est_method
):
    spec = {"xformla": "~" + "+".join(f"cov{j}" for j in range(1, 21)), "est_method": est_method}
    data = preprocess_did(wide_panel, yname="y", tname="time", idname="id", gname="group", weightsname="w", **spec)
    units = data.time_invariant_data["id"]
    cohorts = data.time_invariant_data["group"].to_numpy()
    edges = np.flatnonzero(cohorts[1:] != cohorts[:-1])
    rows = np.r_[0, edges, edges + 1, len(units) - 1]

    result = compute_att_gt(data)
    influence = result.influence_functions.toarray()
    expected = weight_derivative(
        lambda factors: wide_panel_atts_with_scaled_unit_weights(wide_panel, units, factors, **spec), len(units), rows
    )

    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert influence.shape == (len(units), 4)
    assert np.all(np.isfinite(influence))
    np.testing.assert_allclose(influence[rows], expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_wide_cross_section_influence_functions_match_weight_derivative(
    wide_cross_section, wide_cross_section_atts_with_scaled_row_weights, weight_derivative, est_method
):
    spec = {"xformla": "~" + "+".join(f"cov{j}" for j in range(1, 21)), "est_method": est_method}
    data = preprocess_did(
        wide_cross_section, yname="y", tname="time", gname="group", weightsname="w", panel=False, **spec
    )
    original_rows = data.data[".rowid"].to_numpy()
    blocks = data.data.select(["time", "group"]).to_numpy()
    edges = np.flatnonzero(np.any(blocks[1:] != blocks[:-1], axis=1))
    rows = np.r_[0, edges, edges + 1, len(original_rows) - 1]

    result = compute_att_gt(data)
    influence = result.influence_functions.toarray()
    expected = weight_derivative(
        lambda factors: wide_cross_section_atts_with_scaled_row_weights(wide_cross_section, factors, **spec),
        wide_cross_section.height,
        original_rows[rows],
    )

    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    assert influence.shape == (wide_cross_section.height, 4)
    assert np.all(np.isfinite(influence))
    np.testing.assert_allclose(influence[rows], expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
@pytest.mark.parametrize("data_fixture", ["hub_overlap_panel", "hub_overlap_cross_section"])
def test_compute_att_gt_wide_design_trims_comparison_units_at_the_documented_propensity_level(
    request, newton_logit, reference_cell_att, data_fixture, est_method
):
    rows = request.getfixturevalue(data_fixture)(3)
    panel = data_fixture == "hub_overlap_panel"
    covariates = [f"cov{j}" for j in range(1, 21)]
    observations = rows.filter(pl.col("time") == 1) if panel else rows
    treated = (observations["group"] == 2).to_numpy()
    design = np.column_stack([np.ones(observations.height), observations.select(covariates).to_numpy()])
    scores = newton_logit(treated.astype(float), design, observations["w"].to_numpy())
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id" if panel else None,
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method=est_method,
        panel=panel,
    )
    expected = reference_cell_att(rows, covariates, (2, 2, 1, [0]), est_method, panel=panel)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert np.sum(scores[~treated] >= 0.995) == 1
    assert scores.max() < 0.999
    assert [str(warning.message) for warning in caught] == []
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], [expected], rtol=1e-9, atol=0)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
def test_compute_att_gt_wide_design_skips_a_cell_whose_propensity_score_reaches_the_overlap_level(
    hub_overlap_panel, newton_logit, est_method
):
    rows = hub_overlap_panel(1)
    covariates = [f"cov{j}" for j in range(1, 21)]
    first = rows.filter(pl.col("time") == 1)
    design = np.column_stack([np.ones(first.height), first.select(covariates).to_numpy()])
    scores = newton_logit((first["group"] == 2).to_numpy().astype(float), design, first["w"].to_numpy())
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method=est_method,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert scores.max() >= 0.999
    assert result.attgt_list == []
    assert [str(warning.message) for warning in caught] == ["Overlap condition violated for 2 in time period 2"]


@pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
def test_compute_att_gt_wide_panel_skips_cells_whose_comparison_group_has_fewer_units_than_coefficients(
    wide_panel, reference_cell_att
):
    never_treated = wide_panel.filter((pl.col("group") == 0) & (pl.col("time") == 1))["id"].head(20)
    rows = wide_panel.filter((pl.col("group") > 0) | pl.col("id").is_in(never_treated.implode()))
    covariates = [f"cov{j}" for j in range(1, 21)]
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method="reg",
        control_group="notyettreated",
    )
    expected = reference_cell_att(rows, covariates, (2, 2, 1, [0, 3]), "reg")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2)]
    np.testing.assert_allclose(result.attgt_list[0].att, expected, rtol=1e-9, atol=0)
    assert [str(warning.message) for warning in caught] == [
        f"Not enough control units for group {group} in time period {period} to run specified regression"
        for group, period in ((2, 3), (3, 2), (3, 3))
    ]


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_wide_panel_estimates_cells_whose_comparison_group_has_a_few_more_units_than_coefficients(
    wide_panel, reference_cell_att, est_method
):
    never_treated = wide_panel.filter((pl.col("group") == 0) & (pl.col("time") == 1))["id"].head(25)
    rows = wide_panel.filter((pl.col("group") > 0) | pl.col("id").is_in(never_treated.implode()))
    covariates = [f"cov{j}" for j in range(1, 21)]
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method=est_method,
    )
    cells = [(2, 2, 1, [0]), (2, 3, 1, [0]), (3, 2, 1, [0]), (3, 3, 2, [0])]
    expected = [reference_cell_att(rows, covariates, cell, est_method) for cell in cells]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("data_fixture", ["wide_panel", "wide_cross_section"])
@pytest.mark.parametrize("est_method", ["dr", "reg"])
@pytest.mark.parametrize(
    "redundant",
    [pl.col("cov5"), pl.when(pl.col("group") == 0).then(0.0).otherwise(pl.col("cov5"))],
    ids=["copy_of_cov5", "zero_for_comparison_units"],
)
def test_compute_att_gt_wide_design_skips_cells_whose_comparison_covariates_are_singular(
    request, redundant, est_method, data_fixture
):
    rows = request.getfixturevalue(data_fixture).with_columns(redundant.alias("redundant"))
    panel = data_fixture == "wide_panel"
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id" if panel else None,
        gname="group",
        xformla="~" + "+".join([*(f"cov{j}" for j in range(1, 21)), "redundant"]),
        weightsname="w",
        est_method=est_method,
        panel=panel,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert result.attgt_list == []
    assert [str(warning.message) for warning in caught] == [
        f"Not enough control units for group {group} in time period {period} to run specified regression"
        for group, period in ((2, 2), (2, 3), (3, 2), (3, 3))
    ]


@pytest.mark.parametrize("data_fixture", ["wide_panel", "wide_cross_section"])
def test_compute_att_gt_ipw_wide_design_estimates_cells_with_a_covariate_constant_among_comparison_units(
    request, reference_cell_att, data_fixture
):
    rows = request.getfixturevalue(data_fixture).with_columns(
        pl.when(pl.col("group") == 0).then(0.0).otherwise(pl.col("cov5")).alias("treated_cov5")
    )
    panel = data_fixture == "wide_panel"
    covariates = [*(f"cov{j}" for j in range(1, 21)), "treated_cov5"]
    data = preprocess_did(
        rows,
        yname="y",
        tname="time",
        idname="id" if panel else None,
        gname="group",
        xformla="~" + "+".join(covariates),
        weightsname="w",
        est_method="ipw",
        panel=panel,
    )
    cells = [(2, 2, 1, [0]), (2, 3, 1, [0]), (3, 2, 1, [0]), (3, 3, 2, [0])]
    expected = [reference_cell_att(rows, covariates, cell, "ipw", panel=panel) for cell in cells]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], expected, rtol=1e-9, atol=0)


@pytest.mark.parametrize("data_fixture", ["wide_panel", "wide_cross_section"])
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_compute_att_gt_wide_design_with_a_nearly_repeated_covariate_matches_the_fit_with_their_difference(
    request, reference_cell_att, est_method, data_fixture
):
    rows = request.getfixturevalue(data_fixture).with_columns(
        (pl.col("cov5") + 1e-6 * pl.col("noise")).alias("near_cov5")
    )
    panel = data_fixture == "wide_panel"
    covariates = [f"cov{j}" for j in range(1, 21)]
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id" if panel else None,
        "gname": "group",
        "weightsname": "w",
        "est_method": est_method,
        "panel": panel,
    }
    data = preprocess_did(rows, xformla="~" + "+".join([*covariates, "near_cov5"]), **spec)
    separated = compute_att_gt(preprocess_did(rows, xformla="~" + "+".join([*covariates, "noise"]), **spec))
    expected_se = np.sqrt(np.mean(separated.influence_functions.toarray() ** 2, axis=0))
    cells = [(2, 2, 1, [0]), (2, 3, 1, [0]), (3, 2, 1, [0]), (3, 3, 2, [0])]
    expected = [reference_cell_att(rows, [*covariates, "noise"], cell, est_method, panel=panel) for cell in cells]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_att_gt(data)
    se = np.sqrt(np.mean(result.influence_functions.toarray() ** 2, axis=0))

    assert [str(warning.message) for warning in caught] == []
    assert [(cell.group, cell.year) for cell in result.attgt_list] == [(2, 2), (2, 3), (3, 2), (3, 3)]
    np.testing.assert_allclose([cell.att for cell in result.attgt_list], expected, rtol=1e-8, atol=0)
    assert np.all(np.isfinite(se))
    assert np.all(np.isfinite(expected_se))
    np.testing.assert_allclose(se, expected_se, rtol=1e-4, atol=0)
