"""Tests for fitting att_gt on repeated cross-sections and unbalanced panels held in row shards."""

import re
from dataclasses import replace
from itertools import pairwise

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from moderndid import aggte, att_gt, drdid_rc, preprocess_did
from moderndid.core.bootstrap import canonical_keys, keyed_rademacher
from moderndid.did.mboot import _mboot_from_draws
from moderndid.did.shards.container import RowShard
from moderndid.did.shards.layout import row_keys, row_shard_from_arrow
from moderndid.did.shards.local import LocalExecutor, row_shards_from_did_data
from moderndid.did.shards.row_driver import fit_att_gt_on_row_shards


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"weightsname": "w", "clustervars": ["countyreal", "state"]},
        {"anticipation": 1, "clustervars": ["countyreal", "state"]},
        {"anticipation": 1, "weightsname": "w"},
    ],
    ids=["plain", "weights_and_state_clusters", "anticipation_and_state_clusters", "anticipation_and_weights"],
)
def test_fit_att_gt_on_row_shards_matches_att_gt_over_the_option_matrix(
    mpdta_layout,
    att_gt_on_row_shards,
    record_warnings,
    assert_same_fit,
    est_method,
    control_group,
    base_period,
    options,
):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "control_group": control_group,
        "base_period": base_period,
        "random_state": 7,
        **layout,
        **options,
    }
    anticipation = [
        (
            UserWarning,
            "anticipation = 1. Never-treated units (with group status 0 or inf) are assumed to never anticipate "
            "treatment. Anticipation only applies to eventually-treated units.",
        )
    ] * spec.get("anticipation", 0)
    time_varying = "weightsname" in options and "allow_unbalanced_panel" in layout

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    fits = [record_warnings(att_gt_on_row_shards, data, n_shards, shard_seed=n_shards, **spec) for n_shards in (1, 3)]
    texts = [text for _, text in memory_warnings]

    assert any(text.startswith("Time-varying weights detected.") for text in texts) == time_varying
    assert any(text.startswith("The units fall in only") for text in texts) == ("clustervars" in options)
    for sharded, shard_warnings in fits:
        assert anticipation + shard_warnings == memory_warnings
        assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_fit_att_gt_on_row_shards_matches_att_gt_on_generated_data_with_four_covariates(
    scalable_layout, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method
):
    data, layout = scalable_layout
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2 + cov3 + cov4",
        "clustervars": ["cluster"],
        "est_method": est_method,
        "control_group": "notyettreated",
        "base_period": "universal",
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert shard_warnings == memory_warnings
    assert sharded.influence_moments.n_clusters == data["cluster"].n_unique()
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_fit_att_gt_on_row_shards_matches_att_gt_without_covariates(
    mpdta_layout, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method
):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~1",
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


def test_fit_att_gt_on_row_shards_bootstrap_draws_a_keyed_multiplier_for_each_row_and_each_copy_of_a_row(
    mpdta_with_duplicate_rows, att_gt_on_row_shards, splitmix_row_keys
):
    data = mpdta_with_duplicate_rows
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "panel": False,
    }
    did_data = preprocess_did(data, **spec)
    rows = did_data.data
    period = np.searchsorted(did_data.config.time_periods, rows["year"].to_numpy())
    values = [rows["first.treat"], rows["lemp"], rows["w"], np.asarray(did_data.covariates_matrix)[:, 1]]
    keys = splitmix_row_keys(period, values)
    shared_keys = splitmix_row_keys(period, values, copy=np.zeros(rows.height))

    memory = att_gt(data, cband=False, **spec)
    fits = [
        att_gt_on_row_shards(data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec)
        for n_shards in (1, 3)
    ]
    n = memory.n_units
    expected = _mboot_from_draws(
        keyed_rademacher(keys, 20261020, 499).T @ memory.influence_func / np.sqrt(n),
        np.sqrt(n),
        0.05,
        skip_small_scales=True,
    )
    shared = _mboot_from_draws(
        keyed_rademacher(shared_keys, 20261020, 499).T @ memory.influence_func / np.sqrt(n),
        np.sqrt(n),
        0.05,
        skip_small_scales=True,
    )

    assert data.with_columns(pl.col("lemp") + 0.0).is_duplicated().sum() == 120
    assert np.signbit(rows["lemp"].to_numpy()).sum() == 5
    assert len(np.unique(keys)) == n == data.height
    assert np.max(np.abs(shared["se"] - expected["se"])) > 1e-6
    for sharded in fits:
        np.testing.assert_allclose(
            sharded.se_gt, np.where(np.isnan(memory.se_gt), np.nan, expected["se"]), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(sharded.critical_value, expected["crit_val"], rtol=1e-12)
    np.testing.assert_allclose(
        fits[1].influence_moments.bootstrap_sums, fits[0].influence_moments.bootstrap_sums, rtol=1e-12, atol=1e-10
    )


def test_fit_att_gt_on_row_shards_bootstrap_draws_a_keyed_multiplier_for_each_unit_of_an_unbalanced_panel(
    mpdta_thinned, att_gt_on_row_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "allow_unbalanced_panel": True,
    }
    units = preprocess_did(mpdta_thinned, **spec).time_invariant_data["countyreal"]

    memory = att_gt(mpdta_thinned, cband=False, **spec)
    fits = [
        att_gt_on_row_shards(
            mpdta_thinned, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec
        )
        for n_shards in (1, 3)
    ]
    draws = keyed_rademacher(canonical_keys(units.to_numpy()), 20261020, 499).T @ memory.influence_func
    expected = _mboot_from_draws(draws / np.sqrt(len(units)), np.sqrt(len(units)), 0.05, skip_small_scales=True)

    assert memory.n_units == len(units) < mpdta_thinned.height
    for sharded in fits:
        np.testing.assert_allclose(
            sharded.se_gt, np.where(np.isnan(memory.se_gt), np.nan, expected["se"]), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(sharded.critical_value, expected["crit_val"], rtol=1e-12)
    np.testing.assert_allclose(
        fits[1].influence_moments.bootstrap_sums, fits[0].influence_moments.bootstrap_sums, rtol=1e-12, atol=1e-10
    )


def test_fit_att_gt_on_row_shards_bootstrap_draws_a_keyed_multiplier_for_each_cluster(
    mpdta_layout, att_gt_on_row_shards, record_warnings
):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["state"],
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, cband=False, **spec)
    fits = [
        record_warnings(
            att_gt_on_row_shards, data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=11, **spec
        )
        for n_shards in (1, 3)
    ]
    clusters, codes = np.unique(memory.estimation_params["cluster"], return_inverse=True)
    sums = np.column_stack([np.bincount(codes, weights=column) for column in memory.influence_func.T])
    draws = keyed_rademacher(canonical_keys(clusters), 11, 499).T @ sums
    expected = _mboot_from_draws(
        draws / np.sqrt(len(clusters)), memory.n_units / np.sqrt(len(clusters)), 0.05, skip_small_scales=True
    )

    for sharded, shard_warnings in fits:
        assert shard_warnings == memory_warnings
        np.testing.assert_allclose(
            sharded.se_gt, np.where(np.isnan(memory.se_gt), np.nan, expected["se"]), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(sharded.critical_value, expected["crit_val"], rtol=1e-12)
        assert sharded.influence_moments.n_clusters == len(clusters) == 29


def test_fit_att_gt_on_row_shards_moments_sum_the_rows_of_each_unit_beside_its_cohort_weight(
    mpdta_layout, att_gt_on_row_shards, record_warnings
):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["state"],
        **layout,
    }

    memory, _ = record_warnings(att_gt, data, cband=False, **spec)
    sharded, _ = record_warnings(att_gt_on_row_shards, data, 3, boot=True, biters=199, random_state=3, **spec)
    moments = sharded.influence_moments
    cohorts = np.asarray(memory.G, dtype=float)
    weights = np.asarray(memory.weights_ind)
    rows = np.column_stack(
        [memory.influence_func, *(weights * (cohorts == g) for g in (2004, 2006, 2007)), np.ones(memory.n_units)]
    )
    clusters, codes = np.unique(memory.estimation_params["cluster"], return_inverse=True)
    cluster_rows = np.column_stack([np.bincount(codes, weights=column) for column in rows.T])
    cluster_gram = cluster_rows.T @ cluster_rows
    cells = slice(len(memory.att_gt))

    np.testing.assert_array_equal(moments.cohorts, [2004, 2006, 2007])
    np.testing.assert_allclose(moments.gram, rows.T @ rows, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram[cells, cells], cluster_gram[cells, cells], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, cluster_gram, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(
        moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(clusters), 3, 199).T @ cluster_rows,
        rtol=1e-12,
        atol=1e-10,
    )
    assert (moments.n_clusters, moments.random_state) == (29, 3)
    assert sharded.influence_func is None


@pytest.mark.parametrize("aggregation", ["group", "dynamic", "calendar"])
def test_aggte_on_row_shards_matches_aggte_in_memory(mpdta_layout, att_gt_on_row_shards, record_warnings, aggregation):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "random_state": 7,
        **layout,
    }

    memory_fit, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded_fit, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)
    unweighted_fit, _ = record_warnings(att_gt, data, **{**spec, "weightsname": None})
    memory = aggte(memory_fit, type=aggregation, cband=False)
    sharded = aggte(sharded_fit, type=aggregation, cband=False)
    unweighted = aggte(unweighted_fit, type=aggregation, cband=False)

    assert shard_warnings == memory_warnings
    assert abs(memory.overall_att - unweighted.overall_att) > 1e-3
    np.testing.assert_array_equal(sharded.event_times, memory.event_times)
    np.testing.assert_allclose(sharded.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


def test_aggte_simple_on_row_shards_matches_aggte_in_memory(mpdta_layout, att_gt_on_row_shards, record_warnings):
    data, layout = mpdta_layout
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "random_state": 7,
        **layout,
    }

    memory_fit, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded_fit, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)
    memory = aggte(memory_fit, type="simple", cband=False)
    sharded = aggte(sharded_fit, type="simple", cband=False)

    assert shard_warnings == memory_warnings
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("base_period, n_failed", [("varying", 0), ("universal", 9)])
def test_fit_att_gt_on_row_shards_warns_and_drops_cells_whose_groups_miss_a_period_like_att_gt(
    mpdta_layout_with_cohort_gaps,
    att_gt_on_row_shards,
    record_warnings,
    assert_same_fit,
    est_method,
    base_period,
    n_failed,
):
    data, layout = mpdta_layout_with_cohort_gaps
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "base_period": base_period,
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)
    texts = {text for _, text in memory_warnings}

    assert {
        "No treated units in group 2004 in pre-treatment period",
        "No control units for group 2004 in time period 2004",
        "No control units for group 2006 in time period 2004",
    } <= texts
    assert not any("group 2004 in time period 2005" in text for text in texts)
    assert np.isnan(memory.att_gt).sum() == n_failed
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_fit_att_gt_on_row_shards_counts_units_of_one_row_and_units_in_no_kept_cell_like_att_gt(
    mpdta_single_row_units, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method, base_period
):
    data = mpdta_single_row_units
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "allow_unbalanced_panel": True,
        "est_method": est_method,
        "base_period": base_period,
        "random_state": 7,
    }
    rows_per_county = data.group_by("countyreal").agg(pl.len(), pl.col("year").min())

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert (rows_per_county["len"] == 1).sum() > 40
    assert ((rows_per_county["len"] == 1) & (rows_per_county["year"] == 2007)).sum() > 0
    assert (UserWarning, "No control units for group 2007 in time period 2007") in memory_warnings
    assert sharded.n_units == rows_per_county.height
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


def test_row_shards_read_from_chunks_of_the_unit_table_hold_whole_units_and_match_att_gt(
    mpdta_thinned, unit_table, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "allow_unbalanced_panel": True,
        "cband": False,
    }
    did_data = preprocess_did(mpdta_thinned, **spec)
    table = unit_table(did_data)
    shards = [row_shard_from_arrow(table.slice(start, 7), 1) for start in range(0, table.num_rows, 7)]

    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        memory = att_gt(mpdta_thinned, **spec)
    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        sharded = fit_att_gt_on_row_shards(
            LocalExecutor(shards),
            did_data.config,
            did_data.cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )

    assert sum(len(shard.unit_keys) for shard in shards) == table.num_rows == memory.n_units
    assert sum(len(shard.y) for shard in shards) == mpdta_thinned.height
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_fit_att_gt_on_row_shards_fails_regressions_with_fewer_positive_weights_than_coefficients_like_att_gt(
    mpdta_layout_with_one_positive_comparison_weight, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method
):
    data, layout = mpdta_layout_with_one_positive_comparison_weight
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert (
        UserWarning,
        "The 2x2 comparison for group 2004 in time period 2004 failed. The comparison group in the pre-period has "
        "fewer observations with positive weight (1) than regression coefficients (2). Pass fewer covariates.",
    ) in memory_warnings
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


def test_fit_att_gt_on_row_shards_warns_about_extreme_ratios_of_positive_weights_only_like_att_gt(
    mpdta_layout_with_extreme_weight_ratio, att_gt_on_row_shards, record_warnings, assert_same_fit
):
    data, layout = mpdta_layout_with_extreme_weight_ratio
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "random_state": 7,
        **layout,
    }
    extreme = (UserWarning, "Extreme weight ratios detected. Results may be numerically unstable.")

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert 0 < memory_warnings.count(extreme) < len(memory.att_gt)
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_fit_att_gt_on_row_shards_fails_cells_whose_side_has_zero_weight_in_a_period_like_att_gt(
    mpdta_layout_with_zero_weight_periods, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method
):
    data, layout = mpdta_layout_with_zero_weight_periods
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "est_method": est_method,
        "base_period": "universal",
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert {
        (
            UserWarning,
            "The 2x2 comparison for group 2006 in time period 2006 failed. The treated units in time period 2005 all "
            "have zero weight.",
        ),
        (
            UserWarning,
            "The 2x2 comparison for group 2004 in time period 2006 failed. The comparison units in time period 2006 "
            "all have zero weight.",
        ),
    } <= set(memory_warnings)
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("whole_clusters", [False, True], ids=["spread", "whole"])
def test_fit_att_gt_on_row_shards_clusters_rows_or_units_and_warns_about_few_clusters_of_a_cohort_like_att_gt(
    mpdta_layout_with_few_clusters, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method, whole_clusters
):
    data, layout = mpdta_layout_with_few_clusters
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["cluster"],
        "est_method": est_method,
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, whole_clusters=whole_clusters, **spec)

    assert any(
        text.startswith("The treated units of cohort 2004 fall in 3 clusters and those of cohort 2006 in")
        for _, text in memory_warnings
    )
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("whole_clusters", [False, True], ids=["spread", "whole"])
def test_fit_att_gt_on_row_shards_clusters_each_row_of_repeated_cross_sections_by_its_own_cluster(
    mpdta_row_clusters, att_gt_on_row_shards, record_warnings, assert_same_fit, whole_clusters
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["cluster"],
        "panel": False,
        "random_state": 7,
    }

    memory, memory_warnings = record_warnings(att_gt, mpdta_row_clusters, **spec)
    sharded, shard_warnings = record_warnings(
        att_gt_on_row_shards, mpdta_row_clusters, 3, whole_clusters=whole_clusters, **spec
    )
    counties = mpdta_row_clusters.group_by("countyreal").agg(pl.col("cluster").n_unique())

    assert counties["cluster"].min() == 5
    assert sharded.influence_moments.n_clusters == 23
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
@pytest.mark.parametrize("over", ["warn", "ignore"])
def test_fit_att_gt_on_row_shards_replays_the_overflow_warnings_of_every_logit_under_the_error_state(
    far_comparison_unit, att_gt_on_row_shards, record_warnings, assert_same_fit, est_method, over
):
    data, layout = far_comparison_unit
    spec = {
        "yname": "y",
        "tname": "t",
        "gname": "g",
        "xformla": "~x",
        "est_method": est_method,
        "random_state": 7,
        **layout,
    }

    with np.errstate(over=over):
        memory, memory_warnings = record_warnings(att_gt, data, **spec)
        sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)
    overflows = memory_warnings.count((RuntimeWarning, "overflow encountered in exp"))

    assert (overflows > 0) == (over == "warn")
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
def test_fit_att_gt_on_row_shards_raises_the_overflow_of_a_logit_when_the_error_state_raises(
    far_comparison_unit, att_gt_on_row_shards, est_method
):
    data, layout = far_comparison_unit
    spec = {
        "yname": "y",
        "tname": "t",
        "gname": "g",
        "xformla": "~x",
        "est_method": est_method,
        "random_state": 7,
        **layout,
    }

    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="^overflow encountered in exp$"):
        att_gt(data, **spec)
    with np.errstate(over="raise"), pytest.raises(FloatingPointError, match="^overflow encountered in exp$"):
        att_gt_on_row_shards(data, 3, **spec)


@pytest.mark.parametrize(
    "degenerate_comparison_rows",
    [
        (kind, layout)
        for kind in ("constant", "exact", "two_rows")
        for layout in ("repeated_cross_sections", "unbalanced_panel")
    ],
    indirect=True,
    ids=lambda pair: "-".join(pair),
)
def test_reg_on_row_shards_estimates_degenerate_comparison_groups_quietly_like_att_gt(
    degenerate_comparison_rows, att_gt_on_row_shards, record_warnings, assert_same_fit
):
    data, layout = degenerate_comparison_rows
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~x1",
        "est_method": "reg",
        "base_period": "universal",
        "random_state": 7,
        **layout,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert memory_warnings == []
    assert shard_warnings == []
    assert np.all(np.isfinite(memory.att_gt))
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("xformla", ["~lpop + lpop2", "~lpop + big"], ids=["exactly", "up_to_rounding"])
def test_fit_att_gt_on_row_shards_drops_collinear_propensity_covariates_with_the_warning_of_att_gt(
    mpdta_layout_with_collinear_covariates, att_gt_on_row_shards, record_warnings, assert_same_fit, xformla
):
    data, layout = mpdta_layout_with_collinear_covariates
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": xformla,
        "est_method": "ipw",
        "random_state": 7,
        **layout,
    }
    dropped = (UserWarning, "The propensity score model dropped 1 covariate that is collinear with the others.")

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)

    assert memory_warnings.count(dropped) == 12
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


def test_fit_att_gt_on_row_shards_warns_about_a_dropped_propensity_covariate_between_the_screen_and_the_estimator(
    far_comparison_unit, att_gt_on_row_shards, record_warnings, assert_same_fit
):
    data, layout = far_comparison_unit
    data = data.with_columns((2 * pl.col("x")).alias("x2"))
    spec = {
        "yname": "y",
        "tname": "t",
        "gname": "g",
        "xformla": "~x + x2",
        "est_method": "ipw",
        "random_state": 7,
        **layout,
    }
    overflow = (RuntimeWarning, "overflow encountered in exp")

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_row_shards, data, 3, **spec)
    dropped = memory_warnings.index(
        (UserWarning, "The propensity score model dropped 1 covariate that is collinear with the others.")
    )

    assert memory_warnings[dropped - 1] == memory_warnings[dropped + 1] == overflow
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("option", ["boot", "cband"])
def test_fit_att_gt_on_row_shards_rejects_draws_from_rows_without_keys(mpdta_data, att_gt_on_row_shards, option):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "panel": False,
        "boot": False,
        "cband": False,
        option: True,
    }

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The fit draws bootstrap multipliers, but a shard holds rows without keys. Rebuild the row table with "
            "copy indices."
        ),
    ):
        att_gt_on_row_shards(mpdta_data, 3, keyed=False, random_state=7, **spec)


def test_fit_att_gt_on_row_shards_without_draws_needs_no_row_keys(mpdta_data, att_gt_on_row_shards, assert_same_fit):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "panel": False,
        "cband": False,
    }

    memory = att_gt(mpdta_data, **spec)
    sharded = att_gt_on_row_shards(mpdta_data, 3, keyed=False, **spec)

    assert sharded.influence_moments.bootstrap_sums is None
    assert_same_fit(sharded, memory, 1e-12)


def test_fit_att_gt_on_row_shards_draws_by_cluster_without_row_keys(mpdta_with_states, att_gt_on_row_shards):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["state"],
        "panel": False,
        "boot": True,
        "biters": 99,
    }

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        keyed = att_gt_on_row_shards(mpdta_with_states, 3, random_state=7, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        unkeyed = att_gt_on_row_shards(mpdta_with_states, 3, keyed=False, random_state=7, **spec)

    np.testing.assert_array_equal(unkeyed.influence_moments.bootstrap_sums, keyed.influence_moments.bootstrap_sums)
    np.testing.assert_array_equal(unkeyed.se_gt, keyed.se_gt)


@pytest.mark.parametrize(
    "data_fixture, layout, message",
    [
        ("mpdta_data", {"panel": False}, "The shards hold 1666 units but the configuration counts 2500."),
        (
            "mpdta_unbalanced",
            {"idname": "countyreal", "allow_unbalanced_panel": True},
            "The shards hold 333 units but the configuration counts 500.",
        ),
    ],
    ids=["repeated_cross_sections", "unbalanced_panel"],
)
def test_fit_att_gt_on_row_shards_rejects_shards_of_other_data(request, data_fixture, layout, message):
    data = request.getfixturevalue(data_fixture)
    did_data = preprocess_did(data, yname="lemp", tname="year", gname="first.treat", xformla="~lpop", **layout)
    shards = row_shards_from_did_data(did_data, 3, random_state=0)

    with pytest.raises(
        RuntimeError, match=re.escape(f"{message} Build the shards from the panel that set the configuration.")
    ):
        fit_att_gt_on_row_shards(
            LocalExecutor(shards[1:]),
            did_data.config,
            did_data.cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"panel": True},
            "The row engine fits repeated cross-sections and unbalanced panels only. Pass the configuration that "
            "preprocessing sets for panel=False or for a panel in which some unit misses a period.",
        ),
        (
            {"est_method": "probit"},
            "est_method='probit' is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'.",
        ),
        ({"est_method": drdid_rc}, " is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'."),
    ],
    ids=["balanced_panel", "unknown_estimator", "custom_estimator"],
)
def test_fit_att_gt_on_row_shards_rejects_configurations_it_cannot_fit_before_any_pass(mpdta_data, change, message):
    did_data = preprocess_did(mpdta_data, yname="lemp", tname="year", gname="first.treat", panel=False)
    executor = LocalExecutor(row_shards_from_did_data(did_data, 3, random_state=0))

    with pytest.raises(ValueError, match=re.escape(message)):
        fit_att_gt_on_row_shards(
            executor,
            replace(did_data.config, **change),
            did_data.cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )

    assert executor.n_passes == 0


@pytest.mark.parametrize("random_state", [-1, 1.5, True, "7"])
def test_fit_att_gt_on_row_shards_rejects_a_seed_that_a_generator_cannot_take_before_any_pass(mpdta_data, random_state):
    did_data = preprocess_did(mpdta_data, yname="lemp", tname="year", gname="first.treat", panel=False)
    executor = LocalExecutor(row_shards_from_did_data(did_data, 3, random_state=0))

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"random_state={random_state!r} is not valid for a fit on shards. Pass None, a non-negative integer, a "
            "SeedSequence, a BitGenerator, or a Generator."
        ),
    ):
        fit_att_gt_on_row_shards(
            executor, did_data.config, did_data.cohort_counts, random_state, {"bootstrap": False, "uniform_bands": True}
        )

    assert executor.n_passes == 0


@pytest.mark.parametrize(
    "data_fixture, layout",
    [("mpdta_data", {"panel": False}), ("mpdta_unbalanced", {"idname": "countyreal", "allow_unbalanced_panel": True})],
    ids=["repeated_cross_sections", "unbalanced_panel"],
)
def test_fit_att_gt_on_row_shards_with_outcome_regressions_takes_three_passes(request, data_fixture, layout):
    data = request.getfixturevalue(data_fixture)
    did_data = preprocess_did(
        data, yname="lemp", tname="year", gname="first.treat", xformla="~lpop", est_method="reg", **layout
    )
    executor = LocalExecutor(row_shards_from_did_data(did_data, 3, random_state=0))

    result = fit_att_gt_on_row_shards(
        executor, did_data.config, did_data.cohort_counts, 0, {"bootstrap": False, "uniform_bands": False}
    )

    assert len(result.att_gt) == 12
    assert executor.n_passes == 3


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_fit_att_gt_on_row_shards_matches_att_gt_with_empty_and_single_row_shards(
    mpdta_data, assert_same_fit, est_method
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "panel": False,
        "cband": False,
    }
    did_data = preprocess_did(mpdta_data, **spec)
    whole = row_shards_from_did_data(did_data, 1, random_state=0)[0]
    bounds = [0, 0, 1, 2, 2, len(whole.y)]
    shards = [
        RowShard(*(None if field is None else field[start:stop] for field in whole)) for start, stop in pairwise(bounds)
    ]

    memory = att_gt(mpdta_data, **spec)
    sharded = fit_att_gt_on_row_shards(
        LocalExecutor(shards), did_data.config, did_data.cohort_counts, 0, {"bootstrap": False, "uniform_bands": False}
    )

    assert [len(shard.y) for shard in shards] == [0, 1, 1, 0, 2498]
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("n_shards", [1, 3, 7])
def test_row_shards_from_did_data_keys_each_row_of_repeated_cross_sections_by_its_values_and_copy_index(
    mpdta_with_duplicate_rows, splitmix_row_keys, n_shards
):
    did_data = preprocess_did(
        mpdta_with_duplicate_rows,
        yname="lemp",
        tname="year",
        gname="first.treat",
        xformla="~lpop",
        weightsname="w",
        panel=False,
    )
    rows = did_data.data
    period = np.searchsorted(did_data.config.time_periods, rows["year"].to_numpy())
    covariate = np.asarray(did_data.covariates_matrix)[:, 1]
    keys = splitmix_row_keys(period, [rows["first.treat"], rows["lemp"], rows["w"], covariate])

    shards = row_shards_from_did_data(did_data, n_shards, random_state=5)
    sizes = [len(shard.y) for shard in shards]
    order = np.argsort(np.concatenate([shard.unit_keys for shard in shards]))
    expected_order = np.argsort(keys)

    assert np.signbit(rows["lemp"].to_numpy()).sum() == 5
    assert max(sizes) - min(sizes) <= 1
    assert all(shard.units is None for shard in shards)
    np.testing.assert_array_equal(np.concatenate([shard.unit_keys for shard in shards])[order], keys[expected_order])
    np.testing.assert_array_equal(np.concatenate([shard.y for shard in shards])[order], rows["lemp"][expected_order])
    np.testing.assert_array_equal(np.concatenate([shard.period for shard in shards])[order], period[expected_order])
    np.testing.assert_array_equal(
        np.concatenate([shard.covariates for shard in shards])[order][:, 0], covariate[expected_order]
    )


def test_row_shards_from_did_data_puts_each_unit_of_an_unbalanced_panel_whole_and_in_period_order_into_one_shard(
    mpdta_thinned,
):
    did_data = preprocess_did(
        mpdta_thinned,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        clustervars=["state"],
        allow_unbalanced_panel=True,
    )
    counties = mpdta_thinned.group_by("countyreal").agg(pl.len(), pl.col("state").first())
    county_keys = canonical_keys(counties["countyreal"].to_numpy())

    shards = row_shards_from_did_data(did_data, 3, random_state=5, cluster="state")
    keys = np.concatenate([shard.unit_keys for shard in shards])
    order = np.argsort(keys)
    expected_order = np.argsort(county_keys)

    np.testing.assert_array_equal(keys[order], county_keys[expected_order])
    np.testing.assert_array_equal(
        np.concatenate([np.bincount(shard.units) for shard in shards])[order],
        counties["len"].to_numpy()[expected_order],
    )
    np.testing.assert_array_equal(
        np.concatenate([shard.cluster_keys for shard in shards])[order],
        canonical_keys(counties["state"].to_numpy())[expected_order],
    )
    for shard in shards:
        assert np.all(np.diff(shard.units) >= 0)
        assert np.all(np.diff(shard.period)[np.diff(shard.units) == 0] > 0)


def test_row_shards_from_did_data_rejects_a_balanced_panel(mpdta_data):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "row_shards_from_did_data needs repeated cross-sections or an unbalanced panel. Preprocess the data with "
            "panel=False, or with allow_unbalanced_panel=True on a panel in which some unit misses a period."
        ),
    ):
        row_shards_from_did_data(did_data, 3)


def test_row_shards_from_did_data_rejects_a_cluster_that_preprocessing_did_not_keep(mpdta_with_states):
    did_data = preprocess_did(mpdta_with_states, yname="lemp", tname="year", gname="first.treat", panel=False)

    with pytest.raises(
        ValueError,
        match=re.escape("cluster='state' is not valid. Must be one of the clustervars of the preprocessed data, []."),
    ):
        row_shards_from_did_data(did_data, 3, cluster="state")


@pytest.mark.parametrize("n_shards", [0, -1, 1.5, True])
def test_row_shards_from_did_data_rejects_a_number_of_shards_that_is_not_a_positive_integer(mpdta_data, n_shards):
    did_data = preprocess_did(mpdta_data, yname="lemp", tname="year", gname="first.treat", panel=False)

    with pytest.raises(ValueError, match=re.escape(f"n_shards={n_shards!r} is not valid. Must be a positive integer.")):
        row_shards_from_did_data(did_data, n_shards)


def test_row_keys_hash_the_words_of_each_row_with_the_splitmix64_chain(splitmix_row_keys):
    period = np.array([0, 0, 1, 1, 2])
    cohort = np.array([2004.0, 2004.0, np.inf, np.inf, 2006.0])
    y = np.array([0.5, 0.5, -0.0, 0.0, 1e300])
    raw_weights = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    covariates = np.array([[1.0, -2.0], [1.0, -2.0], [0.0, 0.0], [-0.0, 0.0], [7.0, 5.0]])
    copy = np.array([0, 1, 0, 1, 0])

    keys = row_keys(period, cohort, y, raw_weights, covariates, copy)
    without_values = row_keys(period, cohort, y, None, None, copy)

    np.testing.assert_array_equal(keys, splitmix_row_keys(period, [cohort, y, raw_weights, *covariates.T], copy))
    np.testing.assert_array_equal(without_values, splitmix_row_keys(period, [cohort, y], copy))
    assert len(np.unique(keys)) == 5
    assert keys.dtype == np.uint64


def test_row_shard_from_arrow_reads_the_row_table_and_keys_rows_that_carry_copy_indices(splitmix_row_keys):
    table = pa.table(
        {
            "period": [0, 1, 1],
            "cohort": [2004.0, np.inf, np.inf],
            "y": [1.5, -0.0, 0.0],
            "w": [0.5, 1.25, 1.25],
            "raw_w": [2.0, 5.0, 5.0],
            "x": [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]],
            "cluster": ["a", "b", "b"],
            "copy": [0, 0, 1],
        }
    )

    shard = row_shard_from_arrow(table, 2)
    unkeyed = row_shard_from_arrow(table.drop_columns(["raw_w", "copy"]), 2)

    np.testing.assert_array_equal(shard.period, [0, 1, 1])
    np.testing.assert_array_equal(shard.cohort, [2004.0, np.inf, np.inf])
    np.testing.assert_array_equal(shard.y, [1.5, -0.0, 0.0])
    np.testing.assert_array_equal(shard.weights, [0.5, 1.25, 1.25])
    np.testing.assert_array_equal(shard.covariates, [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_equal(shard.cluster_keys, canonical_keys(np.array(["a", "b", "b"])))
    np.testing.assert_array_equal(
        shard.unit_keys,
        splitmix_row_keys(
            [0, 1, 1], [[2004.0, np.inf, np.inf], [1.5, -0.0, 0.0], [2.0, 5.0, 5.0], [1.0, 3.0, 3.0], [2.0, 4.0, 4.0]]
        ),
    )
    assert shard.units is None
    assert unkeyed.unit_keys is None
    np.testing.assert_array_equal(unkeyed.y, shard.y)


def test_row_shard_from_arrow_flattens_the_lists_of_each_unit_of_the_unit_table():
    table = pa.table(
        {
            "unit": [11, 12],
            "cohort": [2004.0, np.inf],
            "period": [[0, 2], [1]],
            "y": [[1.0, 2.0], [3.0]],
            "w": [[0.5, 0.75], [2.0]],
            "x": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6]],
            "cluster": ["a", "b"],
        }
    )

    shard = row_shard_from_arrow(table, 2)

    np.testing.assert_array_equal(shard.period, [0, 2, 1])
    np.testing.assert_array_equal(shard.cohort, [2004.0, 2004.0, np.inf])
    np.testing.assert_array_equal(shard.y, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(shard.weights, [0.5, 0.75, 2.0])
    np.testing.assert_array_equal(shard.covariates, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    np.testing.assert_array_equal(shard.unit_keys, canonical_keys(np.array([11, 12])))
    np.testing.assert_array_equal(shard.units, [0, 0, 1])
    np.testing.assert_array_equal(shard.cluster_keys, canonical_keys(np.array(["a", "b"])))


def test_row_shard_from_arrow_reads_a_unit_table_of_several_batches_like_one_batch():
    batch = pa.record_batch(
        {
            "unit": [1, 2, 3, 4],
            "cohort": [2.0, np.inf, 3.0, np.inf],
            "period": [[0, 1], [1], [0, 1, 2], [2]],
            "y": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0], [7.0]],
            "x": [[0.1, 0.2], [0.3], [0.4, 0.5, 0.6], [0.7]],
            "cluster": [10, 10, 20, 20],
        }
    )
    table = pa.Table.from_batches([batch.slice(0, 1), batch.slice(1, 2), batch.slice(3)])

    from_batch = row_shard_from_arrow(batch, 1)
    from_table = row_shard_from_arrow(table, 1)

    assert table.column("y").num_chunks == 3
    for field in RowShard._fields:
        np.testing.assert_array_equal(getattr(from_table, field), getattr(from_batch, field))
    np.testing.assert_array_equal(from_table.units, [0, 0, 1, 2, 2, 2, 3])


@pytest.mark.parametrize(
    "columns, message",
    [
        (
            {"unit": [1, 2], "cohort": [3.0, np.inf], "period": [[0, 1], [0]], "y": [[1.0, 2.0], [3.0, 4.0]]},
            "The unit table column 'y' must hold a list of the right length in every row. Rebuild the unit table.",
        ),
        (
            {"unit": [1, 2], "cohort": [3.0, np.inf], "period": [[0, 1], None], "y": [[1.0, 2.0], [3.0]]},
            "The unit table column 'period' must hold a list of the right length in every row. Rebuild the unit table.",
        ),
        (
            {"unit": [1], "cohort": [3.0], "period": [[0, 1]], "y": [[1.0, 2.0]], "x": [[1.0, 2.0, 3.0]]},
            "The unit table column 'x' must hold a list of the right length in every row. Rebuild the unit table.",
        ),
        (
            {"period": [0, 1], "cohort": [3.0, np.inf], "y": [1.0, 2.0], "x": [[1.0, 2.0], [3.0]]},
            "The row table column 'x' must hold a list of the right length in every row. Rebuild the row table.",
        ),
    ],
    ids=["short_outcome_list", "null_period_list", "short_covariate_list", "short_row_covariates"],
)
def test_row_shard_from_arrow_rejects_lists_of_the_wrong_length(columns, message):
    table = pa.table(columns)

    with pytest.raises(RuntimeError, match=re.escape(message)):
        row_shard_from_arrow(table, 2)
