"""Tests for fitting att_gt on a balanced panel held in shards of whole units."""

import pickle
import re
import warnings
from dataclasses import replace
from itertools import pairwise

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from moderndid import aggte, att_gt, drdid_panel, preprocess_did
from moderndid.core.bootstrap import canonical_keys, keyed_rademacher
from moderndid.did.mboot import _mboot_from_draws
from moderndid.did.shards import kernels
from moderndid.did.shards.container import CellSumsState, FitState, MomentsState, Shard, SummaryState
from moderndid.did.shards.driver import fit_att_gt_on_shards
from moderndid.did.shards.kernels import cell_sums_kernel, irls_kernel, moments_kernel, summary_kernel
from moderndid.did.shards.layout import shard_from_arrow
from moderndid.did.shards.local import LocalExecutor, shards_from_did_data
from moderndid.did.shards.plan import plan_cells


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("anticipation", [0, 1])
@pytest.mark.parametrize("weightsname", [None, "w"], ids=["unweighted", "time_varying_weights"])
@pytest.mark.parametrize("clustervars", [None, ["countyreal", "state"]], ids=["unclustered", "state_clusters"])
def test_att_gt_on_shards_matches_att_gt_over_the_option_matrix(
    mpdta_varying_weights_and_states,
    att_gt_on_shards,
    record_warnings,
    assert_same_fit,
    est_method,
    control_group,
    base_period,
    anticipation,
    weightsname,
    clustervars,
):
    data = mpdta_varying_weights_and_states
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "control_group": control_group,
        "base_period": base_period,
        "anticipation": anticipation,
        "weightsname": weightsname,
        "clustervars": clustervars,
        "random_state": 7,
    }
    anticipation_warnings = [
        (
            UserWarning,
            "anticipation = 1. Never-treated units (with group status 0 or inf) are assumed to never anticipate "
            "treatment. Anticipation only applies to eventually-treated units.",
        )
    ] * anticipation

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, data, 3, **spec)

    assert memory_warnings == anticipation_warnings + shard_warnings
    assert any(text.startswith("Time-varying weights detected.") for _, text in shard_warnings) == bool(weightsname)
    assert any(text.startswith("The units fall in only") for _, text in shard_warnings) == bool(clustervars)
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_matches_att_gt_on_a_generated_panel_with_four_covariates(
    scalable_panel, att_gt_on_shards, assert_same_fit, est_method, control_group, base_period
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2 + cov3 + cov4",
        "est_method": est_method,
        "control_group": control_group,
        "base_period": base_period,
        "random_state": 7,
    }

    memory = att_gt(scalable_panel, **spec)
    sharded = att_gt_on_shards(scalable_panel, 3, **spec)

    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_with_clusters_of_a_generated_panel(
    scalable_panel, att_gt_on_shards, assert_same_fit, est_method
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1 + cov2 + cov3 + cov4",
        "clustervars": ["cluster"],
        "est_method": est_method,
        "random_state": 7,
    }

    memory = att_gt(scalable_panel, **spec)
    sharded = att_gt_on_shards(scalable_panel, 7, **spec)

    assert_same_fit(sharded, memory, 1e-12)
    assert sharded.influence_moments.n_clusters == 50


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_without_covariates(mpdta_data, att_gt_on_shards, assert_same_fit, est_method):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~1",
        "est_method": est_method,
        "random_state": 7,
    }

    memory = att_gt(mpdta_data, **spec)
    sharded = att_gt_on_shards(mpdta_data, 3, **spec)

    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_matches_att_gt_for_one_or_seven_shards(
    mpdta_data, att_gt_on_shards, assert_same_fit, est_method, n_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "est_method": est_method,
        "random_state": 7,
    }

    memory = att_gt(mpdta_data, **spec)
    sharded = att_gt_on_shards(mpdta_data, n_shards, shard_seed=n_shards, **spec)

    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_with_empty_and_single_unit_shards(
    mpdta_with_states, assert_same_fit, est_method
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "state"],
        "est_method": est_method,
        "cband": False,
    }
    did_data = preprocess_did(mpdta_with_states, allow_unbalanced_panel=False, **spec)
    whole = shards_from_did_data(did_data, 1, random_state=0)[0]
    bounds = [0, 0, 1, 2, 2, 500]
    shards = [
        Shard(*(None if field is None else field[start:stop] for field in whole)) for start, stop in pairwise(bounds)
    ]

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = att_gt(mpdta_with_states, **spec)
    sharded = fit_att_gt_on_shards(
        LocalExecutor(shards), did_data.config, did_data.cohort_counts, 0, dict(memory.estimation_params, cluster=None)
    )

    assert [len(shard.cohort) for shard in shards] == [0, 1, 1, 0, 498]
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        ("overlap_flip_panel", {"xformla": "~x"}, "Overlap condition violated for 3 in time period 3"),
        (
            "overlap_flip_panel",
            {"xformla": "~x", "est_method": "ipw", "base_period": "universal"},
            "Overlap condition violated for 3 in time period 3",
        ),
        (
            "thin_weighted_controls_panel",
            {"weightsname": "w", "est_method": "ipw"},
            "Overlap condition violated for 3 in time period 2",
        ),
        (
            "tiny_weight_overlap_panel",
            {"xformla": "~x", "weightsname": "w", "base_period": "universal"},
            "Overlap condition violated for 4 in time period 4",
        ),
        (
            "zero_weight_controls_panel",
            {"xformla": "~x", "weightsname": "w"},
            "Not enough control units for group 3 in time period 2 to run specified regression",
        ),
        (
            "zero_weight_controls_panel",
            {"xformla": "~x", "weightsname": "w", "est_method": "reg"},
            "Not enough control units for group 3 in time period 3 to run specified regression",
        ),
        (
            "tiny_late_cohort_panel",
            {"control_group": "notyettreated"},
            "The 2x2 comparison for group 3 in time period 4 failed. No effectively control units after trimming. "
            "Cannot compute ATT.",
        ),
        (
            "tiny_late_cohort_panel",
            {"control_group": "notyettreated", "est_method": "ipw", "base_period": "universal"},
            "The 2x2 comparison for group 3 in time period 4 failed. The estimator returned NaN for the ATT.",
        ),
        (
            "zero_weight_cohort_panel",
            {"weightsname": "w", "est_method": "ipw"},
            "The 2x2 comparison for group 3 in time period 2 failed. The treated units all have zero weight.",
        ),
        (
            "zero_weight_cohort_panel",
            {"weightsname": "w", "est_method": "reg", "base_period": "universal"},
            "The 2x2 comparison for group 3 in time period 1 failed. The treated units all have zero weight.",
        ),
        (
            "zero_weight_never_treated_panel",
            {"weightsname": "w", "control_group": "notyettreated"},
            "Not enough control units for group 3 in time period 4 to run specified regression",
        ),
        (
            "zero_weight_never_treated_panel",
            {"weightsname": "w", "control_group": "notyettreated", "est_method": "reg"},
            "Not enough control units for group 4 in time period 5 to run specified regression",
        ),
        (
            "zero_weight_never_treated_panel",
            {"weightsname": "w", "control_group": "notyettreated", "est_method": "ipw", "base_period": "universal"},
            "Overlap condition violated for 4 in time period 5",
        ),
    ],
)
def test_att_gt_on_shards_warns_and_drops_cells_like_att_gt(
    request, att_gt_on_shards, record_warnings, assert_same_fit, data_fixture, options, message
):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "random_state": 7, **options}

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, data, 3, **spec)

    assert message in [text for _, text in memory_warnings]
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "Overlap condition violated for 2004 in time period 2004"),
        ({"est_method": "ipw", "base_period": "universal"}, "Overlap condition violated for 2004 in time period 2007"),
    ],
)
def test_att_gt_on_shards_warns_like_att_gt_when_a_covariate_separates_a_cohort(
    mpdta_separated_2004, att_gt_on_shards, record_warnings, assert_same_fit, options, message
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~size",
        "random_state": 7,
        **options,
    }

    memory, memory_warnings = record_warnings(att_gt, mpdta_separated_2004, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, mpdta_separated_2004, 3, **spec)

    assert message in [text for _, text in memory_warnings]
    assert any(issubclass(category, PerfectSeparationWarning) for category, _ in memory_warnings)
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize(
    "base_period, message",
    [
        ("varying", "The units fall in only 29 clusters. The treated units of cohort 2006 fall in 3 clusters."),
        (
            "universal",
            "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
            "cohort 2006 in 3 clusters.",
        ),
    ],
)
def test_att_gt_on_shards_warns_about_few_clusters_only_for_cohorts_that_keep_cells(
    mpdta_separated_2004, att_gt_on_shards, record_warnings, assert_same_fit, base_period, message
):
    data = mpdta_separated_2004.with_columns((pl.col("countyreal") // 1000).alias("state"))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~size",
        "clustervars": ["countyreal", "state"],
        "base_period": base_period,
        "random_state": 7,
    }

    memory, memory_warnings = record_warnings(att_gt, data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, data, 3, **spec)

    assert any(text.startswith(message) for _, text in memory_warnings)
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("est_method, base_period", [("dr", "varying"), ("reg", "universal")])
def test_att_gt_on_shards_warns_like_att_gt_when_regression_covariates_are_collinear(
    mpdta_collinear_covariates, att_gt_on_shards, assert_same_fit, est_method, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop + lpop2",
        "est_method": est_method,
        "base_period": base_period,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(mpdta_collinear_covariates, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(mpdta_collinear_covariates, 3, **spec)

    assert "Not enough control units for group 2007 in time period 2007 to run specified regression" in [
        str(w.message) for w in memory_warnings
    ]
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    assert_same_fit(sharded, memory, 1e-10)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_matches_att_gt_when_propensity_covariates_are_collinear(
    mpdta_collinear_covariates, att_gt_on_shards, record_warnings, assert_same_fit, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop + lpop2",
        "est_method": "ipw",
        "base_period": base_period,
        "random_state": 7,
    }
    dropped = (UserWarning, "The propensity score model dropped 1 covariate that is collinear with the others.")

    memory, memory_warnings = record_warnings(att_gt, mpdta_collinear_covariates, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, mpdta_collinear_covariates, 3, **spec)

    assert memory_warnings == [dropped] * 12
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_drops_a_propensity_covariate_collinear_up_to_rounding_like_att_gt(
    mpdta_rounded_collinear_covariate, att_gt_on_shards, record_warnings, assert_same_fit, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop + big",
        "est_method": "ipw",
        "base_period": base_period,
        "random_state": 7,
    }
    dropped = (UserWarning, "The propensity score model dropped 1 covariate that is collinear with the others.")

    memory, memory_warnings = record_warnings(att_gt, mpdta_rounded_collinear_covariate, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, mpdta_rounded_collinear_covariate, 3, **spec)

    assert memory_warnings == [dropped] * 12
    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize(
    "degenerate_comparison_panel, control_group, expected_warnings",
    [
        ("constant_change", "nevertreated", []),
        ("exact_change", "nevertreated", []),
        (
            "two_comparison_units",
            "notyettreated",
            [(UserWarning, "Be aware that there are some small groups in your dataset.\nCheck groups: inf")],
        ),
    ],
    ids=["constant_change", "exact_change", "two_comparison_units"],
    indirect=["degenerate_comparison_panel"],
)
def test_att_gt_on_shards_estimates_degenerate_comparison_groups_of_reg_quietly_like_att_gt(
    degenerate_comparison_panel, att_gt_on_shards, record_warnings, assert_same_fit, control_group, expected_warnings
):
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~x1",
        "weightsname": "w",
        "est_method": "reg",
        "control_group": control_group,
        "base_period": "universal",
        "random_state": 7,
    }

    memory, memory_warnings = record_warnings(att_gt, degenerate_comparison_panel, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, degenerate_comparison_panel, 3, **spec)

    assert memory_warnings == expected_warnings
    assert shard_warnings == memory_warnings
    assert np.all(np.isfinite(memory.att_gt))
    assert_same_fit(sharded, memory, 1e-12)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_bootstrap_finishes_keyed_draws_of_the_influence_function(
    mpdta_data, att_gt_on_shards, base_period, n_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": base_period,
    }
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]

    memory = att_gt(mpdta_data, cband=False, **spec)
    sharded = att_gt_on_shards(
        mpdta_data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec
    )
    draws = keyed_rademacher(canonical_keys(units.to_numpy()), 20261020, 499).T @ memory.influence_func
    expected = _mboot_from_draws(draws / np.sqrt(memory.n_units), np.sqrt(memory.n_units), 0.05, skip_small_scales=True)

    np.testing.assert_allclose(
        sharded.se_gt, np.where(np.isnan(memory.se_gt), np.nan, expected["se"]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(sharded.critical_value, expected["crit_val"], rtol=1e-12)


@pytest.mark.parametrize("dtype", [pl.Int64, pl.String])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_bootstrap_draws_one_keyed_multiplier_for_each_cluster(
    mpdta_with_states, att_gt_on_shards, dtype, n_shards
):
    data = mpdta_with_states.with_columns(pl.col("state").cast(dtype))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "state"],
    }
    units = preprocess_did(data, allow_unbalanced_panel=False, **spec).time_invariant_data

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = att_gt(data, cband=False, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = att_gt_on_shards(
            data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec
        )
    clusters, codes = np.unique(units["state"].to_numpy(), return_inverse=True)
    sums = np.column_stack([np.bincount(codes, weights=column) for column in memory.influence_func.T])
    draws = keyed_rademacher(canonical_keys(clusters), 20261020, 499).T @ sums
    expected = _mboot_from_draws(
        draws / np.sqrt(len(clusters)), memory.n_units / np.sqrt(len(clusters)), 0.05, skip_small_scales=True
    )

    np.testing.assert_allclose(
        sharded.se_gt, np.where(np.isnan(memory.se_gt), np.nan, expected["se"]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(sharded.critical_value, expected["crit_val"], rtol=1e-12)


@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_moments_sum_the_unit_rows_of_the_influence_function(
    mpdta_with_states, att_gt_on_shards, n_shards
):
    data = mpdta_with_states.with_columns(pl.col("lpop").exp().alias("pop"))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "pop",
        "clustervars": ["countyreal", "state"],
    }
    units = preprocess_did(data, allow_unbalanced_panel=False, **spec).time_invariant_data

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = att_gt(data, cband=False, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = att_gt_on_shards(data, n_shards, shard_seed=n_shards, boot=True, biters=199, random_state=3, **spec)
    moments = sharded.influence_moments
    cohorts = np.asarray(memory.G, dtype=float)
    weights = np.asarray(memory.weights_ind)
    rows = np.column_stack(
        [memory.influence_func, *(weights * (cohorts == g) for g in (2004, 2006, 2007)), np.ones(memory.n_units)]
    )
    clusters, codes = np.unique(units["state"].to_numpy(), return_inverse=True)
    cluster_rows = np.column_stack([np.bincount(codes, weights=column) for column in rows.T])

    np.testing.assert_array_equal(moments.cohorts, [2004, 2006, 2007])
    np.testing.assert_allclose(moments.gram, rows.T @ rows, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, cluster_rows.T @ cluster_rows, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(clusters), 3, 199).T @ cluster_rows,
        rtol=1e-12,
        atol=1e-10,
    )
    assert (moments.n_clusters, moments.random_state) == (29, 3)
    assert sharded.influence_func is None
    assert sharded.G is None
    assert sharded.weights_ind is None


def test_att_gt_on_shards_moments_cover_a_shard_of_thousands_of_units(large_scalable_panel, att_gt_on_shards):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
    }
    units = preprocess_did(large_scalable_panel, allow_unbalanced_panel=False, **spec).time_invariant_data

    memory = att_gt(large_scalable_panel, cband=False, **spec)
    sharded = att_gt_on_shards(large_scalable_panel, 1, boot=True, biters=99, random_state=3, **spec)
    moments = sharded.influence_moments
    cohorts = np.asarray(memory.G, dtype=float)
    rows = np.column_stack([memory.influence_func, cohorts == 2, cohorts == 3, np.ones(memory.n_units)])
    clusters, codes = np.unique(units["cluster"].to_numpy(), return_inverse=True)
    cluster_rows = np.column_stack([np.bincount(codes, weights=column) for column in rows.T])

    assert memory.n_units == 2500
    np.testing.assert_allclose(moments.gram, rows.T @ rows, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.cluster_gram, cluster_rows.T @ cluster_rows, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(clusters), 3, 99).T @ cluster_rows,
        rtol=1e-12,
        atol=1e-10,
    )


@pytest.mark.parametrize("biters", [99, 1025, 2100])
def test_att_gt_on_shards_bootstrap_sums_hold_the_keyed_draws_of_every_block_of_draws(
    mpdta_data, att_gt_on_shards, biters
):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"}
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]

    memory = att_gt(mpdta_data, cband=False, **spec)
    sharded = att_gt_on_shards(mpdta_data, 3, boot=True, biters=biters, random_state=5, **spec)
    cohorts = np.asarray(memory.G, dtype=float)
    rows = np.column_stack(
        [memory.influence_func, *(cohorts == g for g in (2004, 2006, 2007)), np.ones(memory.n_units)]
    )

    assert sharded.influence_moments.bootstrap_sums.shape == (biters, rows.shape[1])
    np.testing.assert_allclose(
        sharded.influence_moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(units.to_numpy()), 5, biters).T @ rows,
        rtol=1e-12,
        atol=1e-10,
    )


def test_att_gt_on_shards_derives_its_seed_from_a_seed_sequence_a_bit_generator_or_a_large_integer(
    mpdta_data, att_gt_on_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "boot": True,
        "biters": 99,
    }
    expected = [
        int(np.random.default_rng(seed).integers(2**64, dtype=np.uint64))
        for seed in (np.random.SeedSequence(3), np.random.PCG64(4), 2**64 + 5)
    ]

    fits = [
        att_gt_on_shards(mpdta_data, 3, random_state=seed, **spec)
        for seed in (np.random.SeedSequence(3), np.random.PCG64(4), 2**64 + 5)
    ]
    event_study = aggte(fits[0], type="dynamic")

    assert [fit.influence_moments.random_state for fit in fits] == expected
    assert [fit.estimation_params["random_state"] for fit in fits] == expected
    assert event_study.estimation_params["random_state"] == expected[0]


def test_att_gt_on_shards_stores_the_seed_it_draws_from_a_generator(mpdta_data, att_gt_on_shards):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 99,
    }

    first = att_gt_on_shards(mpdta_data, 3, random_state=np.random.default_rng(3), **spec)
    again = att_gt_on_shards(mpdta_data, 7, shard_seed=1, random_state=np.random.default_rng(3), **spec)
    other = att_gt_on_shards(mpdta_data, 3, random_state=np.random.default_rng(4), **spec)
    event_study = aggte(first, type="dynamic")
    seed = first.influence_moments.random_state

    assert isinstance(seed, int)
    assert 0 <= seed < 2**64
    assert first.estimation_params["random_state"] == seed
    assert event_study.estimation_params["random_state"] == seed
    assert again.influence_moments.random_state == seed
    assert other.influence_moments.random_state != seed
    np.testing.assert_allclose(again.se_gt, first.se_gt, rtol=1e-12)


@pytest.mark.parametrize(
    "data_fixture, columns",
    [
        (
            "mpdta_data",
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"},
        ),
        (
            "mpdta_pop_weighted",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "weightsname": "pop",
            },
        ),
        (
            "scalable_panel",
            {
                "yname": "y",
                "tname": "time",
                "idname": "id",
                "gname": "group",
                "xformla": "~cov1 + cov2 + cov3 + cov4",
                "clustervars": ["cluster"],
            },
        ),
    ],
    ids=["mpdta", "weighted", "clustered"],
)
@pytest.mark.parametrize(
    "options",
    [
        {"type": "group"},
        {"type": "group", "max_e": 1},
        {"type": "dynamic"},
        {"type": "dynamic", "min_e": -2, "max_e": 2},
        {"type": "dynamic", "balance_e": 1},
        {"type": "dynamic", "clustervars": []},
        {"type": "calendar"},
    ],
    ids=["group", "group_max_e", "dynamic", "dynamic_window", "dynamic_balance_e", "dynamic_by_unit", "calendar"],
)
def test_aggte_on_shards_matches_aggte_in_memory(request, att_gt_on_shards, data_fixture, columns, options):
    data = request.getfixturevalue(data_fixture)
    spec = {**columns, "random_state": 7}

    memory = aggte(att_gt(data, **spec), cband=False, **options)
    sharded = aggte(att_gt_on_shards(data, 3, **spec), cband=False, **options)

    np.testing.assert_array_equal(sharded.event_times, memory.event_times)
    np.testing.assert_allclose(sharded.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


@pytest.mark.parametrize(
    "data_fixture, columns",
    [
        (
            "mpdta_data",
            {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"},
        ),
        (
            "mpdta_pop_weighted",
            {
                "yname": "lemp",
                "tname": "year",
                "idname": "countyreal",
                "gname": "first.treat",
                "xformla": "~lpop",
                "weightsname": "pop",
            },
        ),
        (
            "scalable_panel",
            {
                "yname": "y",
                "tname": "time",
                "idname": "id",
                "gname": "group",
                "xformla": "~cov1 + cov2 + cov3 + cov4",
                "clustervars": ["cluster"],
            },
        ),
    ],
    ids=["mpdta", "weighted", "clustered"],
)
@pytest.mark.parametrize("max_e", [np.inf, 1])
def test_aggte_simple_on_shards_matches_aggte_in_memory(request, att_gt_on_shards, data_fixture, columns, max_e):
    data = request.getfixturevalue(data_fixture)
    spec = {**columns, "random_state": 7}

    memory = aggte(att_gt(data, **spec), type="simple", max_e=max_e, cband=False)
    sharded = aggte(att_gt_on_shards(data, 3, **spec), type="simple", max_e=max_e, cband=False)

    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12)


@pytest.mark.parametrize("aggregation", ["group", "dynamic", "calendar"])
def test_aggte_on_shards_weighs_cohorts_by_their_first_period_weights_like_aggte_in_memory(
    mpdta_varying_weights, att_gt_on_shards, aggregation
):
    data = mpdta_varying_weights
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "random_state": 7,
    }

    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        memory = aggte(att_gt(data, weightsname="w", **spec), type=aggregation, cband=False)
    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        sharded = aggte(att_gt_on_shards(data, 3, weightsname="w", **spec), type=aggregation, cband=False)
    unweighted = aggte(att_gt(data, **spec), type=aggregation, cband=False)

    assert data.group_by("countyreal").agg(pl.col("w").n_unique())["w"].min() == 5
    np.testing.assert_array_equal(sharded.event_times, memory.event_times)
    np.testing.assert_allclose(sharded.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)
    assert abs(memory.overall_att - unweighted.overall_att) > 1e-3


def test_aggte_simple_on_shards_weighs_cohorts_by_their_first_period_weights_like_aggte_in_memory(
    mpdta_varying_weights, att_gt_on_shards
):
    data = mpdta_varying_weights
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "random_state": 7,
    }

    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        memory = aggte(att_gt(data, weightsname="w", **spec), type="simple", cband=False)
    with pytest.warns(UserWarning, match="^Time-varying weights detected"):
        sharded = aggte(att_gt_on_shards(data, 3, weightsname="w", **spec), type="simple", cband=False)
    unweighted = aggte(att_gt(data, **spec), type="simple", cband=False)

    assert data.group_by("countyreal").agg(pl.col("w").n_unique())["w"].min() == 5
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12)
    assert abs(memory.overall_att - unweighted.overall_att) > 1e-3


@pytest.mark.parametrize("aggregation", ["group", "dynamic", "calendar"])
def test_aggte_on_shards_leaves_out_failed_cells_like_aggte_in_memory(
    mpdta_separated_2004, att_gt_on_shards, aggregation
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~size",
        "base_period": "universal",
        "random_state": 7,
    }

    with pytest.warns(UserWarning, match="^Overlap condition violated for 2004 in time period"):
        fit = att_gt(mpdta_separated_2004, **spec)
    with pytest.warns(UserWarning, match="^Overlap condition violated for 2004 in time period"):
        sharded_fit = att_gt_on_shards(mpdta_separated_2004, 3, **spec)
    memory = aggte(fit, type=aggregation, na_rm=True, cband=False)
    sharded = aggte(sharded_fit, type=aggregation, na_rm=True, cband=False)

    assert np.isnan(sharded_fit.att_gt).sum() == 4
    np.testing.assert_array_equal(sharded.event_times, memory.event_times)
    np.testing.assert_allclose(sharded.att_by_event, memory.att_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


@pytest.mark.parametrize("n_shards", [1, 7])
def test_aggte_bootstrap_on_shards_finishes_keyed_draws_of_the_event_study(mpdta_data, att_gt_on_shards, n_shards):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": "universal",
    }
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]

    memory = aggte(att_gt(mpdta_data, cband=False, **spec), type="dynamic", cband=False)
    sharded_fit = att_gt_on_shards(
        mpdta_data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec
    )
    sharded = aggte(sharded_fit, type="dynamic", biters=250)
    signs = keyed_rademacher(canonical_keys(units.to_numpy()), 20261020, 250).T / np.sqrt(len(units))
    by_event = _mboot_from_draws(signs @ memory.influence_func, np.sqrt(len(units)), 0.05)
    overall = _mboot_from_draws(signs @ memory.influence_func_overall[:, None], np.sqrt(len(units)), 0.05)

    np.testing.assert_allclose(sharded.se_by_event, by_event["se"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, by_event["crit_val"], rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, overall["se"][0], rtol=1e-12)


def test_aggte_on_a_clustered_shard_fit_draws_its_band_from_the_stored_cluster_draws(
    mpdta_with_states, att_gt_on_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "state"],
    }
    units = preprocess_did(mpdta_with_states, allow_unbalanced_panel=False, **spec).time_invariant_data

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = aggte(att_gt(mpdta_with_states, cband=False, **spec), type="dynamic", cband=False)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded_fit = att_gt_on_shards(mpdta_with_states, 7, biters=399, random_state=11, **spec)
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        sharded = aggte(sharded_fit, type="dynamic")
    clusters, codes = np.unique(units["state"].to_numpy(), return_inverse=True)
    sums = np.column_stack([np.bincount(codes, weights=column) for column in memory.influence_func.T])
    draws = keyed_rademacher(canonical_keys(clusters), 11, 399).T @ sums / np.sqrt(len(clusters))
    band = _mboot_from_draws(draws, len(units) / np.sqrt(len(clusters)), 0.05)

    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, band["crit_val"], rtol=1e-12)
    assert band["crit_val"] > 2


def test_aggte_on_shards_rejects_more_draws_than_the_fit_stored(mpdta_data, att_gt_on_shards):
    sharded = att_gt_on_shards(
        mpdta_data,
        3,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=True,
        biters=99,
        random_state=7,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "biters=100 is more than the 99 bootstrap draws that this att_gt fit on a Spark DataFrame stored. "
            "Pass biters=99 or fewer. For more draws, rerun att_gt with biters=100."
        ),
    ):
        aggte(sharded, type="dynamic", biters=100)


def test_aggte_on_shards_rejects_a_seed_other_than_the_stored_one(mpdta_data, att_gt_on_shards):
    sharded = att_gt_on_shards(
        mpdta_data,
        3,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=True,
        biters=99,
        random_state=7,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "random_state=8 is not the seed 7 of the bootstrap draws that this att_gt fit on a Spark DataFrame "
            "stored. Pass random_state=7 or leave it out."
        ),
    ):
        aggte(sharded, type="dynamic", random_state=8)


def test_aggte_on_a_clustered_shard_fit_rejects_unit_level_draws(mpdta_with_states, att_gt_on_shards):
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = att_gt_on_shards(
            mpdta_with_states,
            3,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            clustervars=["countyreal", "state"],
            boot=True,
            biters=99,
            random_state=7,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "clustervars=[] asks for unit-level bootstrap draws. This att_gt fit on a Spark DataFrame drew "
            "one multiplier for each cluster and holds no draws for single units."
        ),
    ):
        aggte(sharded, type="dynamic", clustervars=[])


def test_aggte_on_a_clustered_shard_fit_rejects_clusters_the_fit_did_not_use(mpdta_with_states, att_gt_on_shards):
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = att_gt_on_shards(
            mpdta_with_states,
            3,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            clustervars=["countyreal", "state"],
            random_state=7,
        )

    with pytest.raises(ValueError, match=re.escape("The att_gt fit did not cluster on 'year'.")):
        aggte(sharded, type="dynamic", clustervars=["year"], cband=False)


@pytest.mark.parametrize("option", ["boot", "cband"])
def test_aggte_on_shards_without_stored_draws_rejects_the_bootstrap(mpdta_data, att_gt_on_shards, option):
    sharded = att_gt_on_shards(
        mpdta_data, 3, yname="lemp", tname="year", idname="countyreal", gname="first.treat", cband=False
    )

    assert sharded.influence_moments.bootstrap_sums is None
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{option}=True needs bootstrap draws that this att_gt fit on a Spark DataFrame did not store. "
            "Pass boot=False and cband=False for analytic standard errors."
        ),
    ):
        aggte(sharded, type="dynamic", **{option: True})


@pytest.mark.parametrize(
    "data_fixture, columns",
    [
        ("mpdta_data", {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}),
        ("scalable_panel", {"yname": "y", "tname": "time", "idname": "id", "gname": "group", "xformla": "~cov1"}),
    ],
    ids=["mpdta", "generated"],
)
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_fit_att_gt_on_shards_with_outcome_regressions_takes_three_passes(
    request, data_fixture, columns, control_group
):
    data = request.getfixturevalue(data_fixture)
    did_data = preprocess_did(
        data, allow_unbalanced_panel=False, est_method="reg", control_group=control_group, **columns
    )
    executor = LocalExecutor(shards_from_did_data(did_data, 3, random_state=0))

    result = fit_att_gt_on_shards(
        executor, did_data.config, did_data.cohort_counts, 0, {"bootstrap": False, "uniform_bands": False}
    )

    assert len(result.att_gt) == 12
    assert executor.n_passes == 3


@pytest.mark.parametrize(
    "change, message",
    [
        (
            {"panel": False},
            "fit_att_gt_on_shards fits balanced panels only. Fit repeated cross-sections and unbalanced panels "
            "with fit_att_gt_on_row_shards.",
        ),
        (
            {"est_method": "probit"},
            "est_method='probit' is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'.",
        ),
        ({"est_method": drdid_panel}, " is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'."),
    ],
    ids=["repeated_cross_sections", "unknown_estimator", "custom_estimator"],
)
def test_fit_att_gt_on_shards_rejects_configurations_it_cannot_fit(mpdta_data, change, message):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )
    executor = LocalExecutor(shards_from_did_data(did_data, 3, random_state=0))

    with pytest.raises(ValueError, match=re.escape(message)):
        fit_att_gt_on_shards(
            executor,
            replace(did_data.config, **change),
            did_data.cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )


def test_fit_att_gt_on_shards_rejects_shards_of_another_panel(mpdta_data):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )
    shards = shards_from_did_data(did_data, 3, random_state=0)

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The shards hold 333 units but the configuration counts 500. "
            "Build the shards from the panel that set the configuration."
        ),
    ):
        fit_att_gt_on_shards(
            LocalExecutor(shards[1:]),
            did_data.config,
            did_data.cohort_counts,
            0,
            {"bootstrap": False, "uniform_bands": False},
        )


@pytest.mark.parametrize("random_state", [-1, 1.5, True, "7"])
def test_fit_att_gt_on_shards_rejects_a_seed_that_a_generator_cannot_take_before_any_pass(mpdta_data, random_state):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )
    executor = LocalExecutor(shards_from_did_data(did_data, 3, random_state=0))

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"random_state={random_state!r} is not valid for a fit on shards. Pass None, a non-negative integer, a "
            "SeedSequence, a BitGenerator, or a Generator."
        ),
    ):
        fit_att_gt_on_shards(
            executor,
            did_data.config,
            did_data.cohort_counts,
            random_state,
            {"bootstrap": False, "uniform_bands": True},
        )

    assert executor.n_passes == 0


@pytest.mark.parametrize("n_shards", [1, 3, 7])
def test_shards_from_did_data_puts_each_unit_whole_into_one_shard(mpdta_with_states, n_shards):
    did_data = preprocess_did(
        mpdta_with_states,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        clustervars=["countyreal", "state"],
        allow_unbalanced_panel=False,
    )
    outcomes = mpdta_with_states.pivot(on="year", index="countyreal", values="lemp", sort_columns=True)
    covariates = mpdta_with_states.pivot(on="year", index="countyreal", values="lpop", sort_columns=True)
    counties = mpdta_with_states.filter(pl.col("year") == 2003).sort("countyreal")

    shards = shards_from_did_data(did_data, n_shards, random_state=5)
    keys = np.concatenate([shard.unit_keys for shard in shards])
    order = np.argsort(keys)
    sizes = [len(shard.unit_keys) for shard in shards]

    assert len(shards) == n_shards
    assert max(sizes) - min(sizes) <= 1
    assert all(shard.weights is None for shard in shards)
    np.testing.assert_array_equal(keys[order], canonical_keys(counties["countyreal"].to_numpy()))
    np.testing.assert_array_equal(
        np.concatenate([shard.y for shard in shards])[order], outcomes.sort("countyreal").drop("countyreal")
    )
    np.testing.assert_array_equal(
        np.concatenate([shard.covariates for shard in shards])[order][:, :, 0],
        covariates.sort("countyreal").drop("countyreal"),
    )
    np.testing.assert_array_equal(
        np.concatenate([shard.cohort for shard in shards])[order],
        counties["first.treat"].cast(pl.Float64).replace(0.0, np.inf),
    )
    np.testing.assert_array_equal(
        np.concatenate([shard.cluster_keys for shard in shards])[order], canonical_keys(counties["state"].to_numpy())
    )


@pytest.mark.parametrize("n_shards", [0, -1, 1.5, True])
def test_shards_from_did_data_rejects_a_number_of_shards_that_is_not_a_positive_integer(mpdta_data, n_shards):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )

    with pytest.raises(ValueError, match=re.escape(f"n_shards={n_shards!r} is not valid. Must be a positive integer.")):
        shards_from_did_data(did_data, n_shards)


def test_shards_from_did_data_rejects_repeated_cross_sections(mpdta_data):
    with pytest.warns(UserWarning, match="^panel=False was specified, but units appear across all time periods"):
        did_data = preprocess_did(
            mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            panel=False,
            allow_unbalanced_panel=False,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "shards_from_did_data needs a balanced panel. Build the shards of repeated cross-sections and "
            "unbalanced panels with row_shards_from_did_data."
        ),
    ):
        shards_from_did_data(did_data, 3)


def test_shards_from_did_data_rejects_a_panel_in_which_some_unit_misses_a_period(mpdta_unbalanced):
    did_data = preprocess_did(
        mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "shards_from_did_data needs a balanced panel. Build the shards of repeated cross-sections and "
            "unbalanced panels with row_shards_from_did_data."
        ),
    ):
        shards_from_did_data(did_data, 3)


def test_att_gt_on_shards_matches_att_gt_on_a_balanced_panel_preprocessed_with_allow_unbalanced_panel(
    mpdta_data, att_gt_on_shards, record_warnings, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "allow_unbalanced_panel": True,
        "random_state": 7,
    }

    memory, memory_warnings = record_warnings(att_gt, mpdta_data, **spec)
    sharded, shard_warnings = record_warnings(att_gt_on_shards, mpdta_data, 3, **spec)

    assert shard_warnings == memory_warnings
    assert_same_fit(sharded, memory, 1e-12)


def test_shard_from_arrow_reads_each_list_column_period_by_period():
    table = pa.table(
        {
            "unit": [11, 12],
            "cohort": [2004.0, np.inf],
            "y": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "w": [[1.0, 1.0, 2.0], [0.5, 0.5, 0.5]],
            "x": [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]],
            "cluster": ["a", "b"],
        }
    )

    shard = shard_from_arrow(table, 3, 2)

    np.testing.assert_array_equal(shard.unit_keys, canonical_keys(np.array([11, 12])))
    np.testing.assert_array_equal(shard.cohort, [2004.0, np.inf])
    np.testing.assert_array_equal(shard.y, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(shard.weights, [[1.0, 1.0, 2.0], [0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(shard.covariates[0], [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(shard.covariates[1], [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]])
    np.testing.assert_array_equal(shard.cluster_keys, canonical_keys(np.array(["a", "b"])))
    assert shard.y.dtype == shard.cohort.dtype == np.float64


def test_shard_from_arrow_leaves_out_the_columns_the_table_lacks():
    table = pa.table({"unit": ["a", "b", "c"], "cohort": [3, 0, 3], "y": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]})

    shard = shard_from_arrow(table, 2, 0)

    assert shard.weights is None
    assert shard.covariates is None
    assert shard.cluster_keys is None
    assert shard.y.shape == (3, 2)
    np.testing.assert_array_equal(shard.unit_keys, canonical_keys(np.array(["a", "b", "c"])))
    np.testing.assert_array_equal(shard.cohort, [3.0, 0.0, 3.0])


def test_shard_from_arrow_reads_a_table_of_several_batches_like_one_batch():
    batch = pa.record_batch(
        {
            "unit": [1, 2, 3, 4],
            "cohort": [2.0, np.inf, 3.0, np.inf],
            "y": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            "x": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
            "cluster": [10, 10, 20, 20],
        }
    )
    table = pa.Table.from_batches([batch.slice(0, 1), batch.slice(1, 2), batch.slice(3)])

    from_batch = shard_from_arrow(batch, 2, 1)
    from_table = shard_from_arrow(table, 2, 1)

    assert table.column("y").num_chunks == 3
    for field in Shard._fields:
        np.testing.assert_array_equal(getattr(from_table, field), getattr(from_batch, field))
    np.testing.assert_array_equal(from_table.covariates[:, :, 0], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])


@pytest.mark.parametrize("outcomes", [[[1.0, 2.0], [3.0]], [[1.0, 2.0], None]], ids=["short_list", "null_list"])
def test_shard_from_arrow_rejects_a_list_column_without_one_value_per_period(outcomes):
    table = pa.table({"unit": [1, 2], "cohort": [3.0, np.inf], "y": outcomes})

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The unit table column 'y' must hold 2 values in every row. Rebuild the unit table from the panel."
        ),
    ):
        shard_from_arrow(table, 2, 0)


def test_shard_kernels_pickle_by_reference_with_their_states_and_statistics(mpdta_data):
    did_data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        allow_unbalanced_panel=False,
    )
    plan = plan_cells(did_data.config, did_data.cohort_counts)
    shard = shards_from_did_data(did_data, 1, random_state=0)[0]
    n_fits, n_cells = len(plan.fits.group), len(plan.cells.fit)
    fits = FitState(
        table=plan.fits, normalizer=np.ones(n_fits), scale=np.ones((n_fits, 2)), coefficients=np.zeros((n_fits, 2))
    )
    passes = [
        (summary_kernel, SummaryState(fits=plan.fits, cells=plan.cells, logit=True, regression=True)),
        (irls_kernel, fits),
        (cell_sums_kernel, CellSumsState(fits=fits, cells=plan.cells, regression=np.zeros((n_cells, 2)), method="dr")),
        (
            moments_kernel,
            MomentsState(
                fits=fits,
                cells=plan.cells,
                kind=np.zeros(n_cells, dtype=np.int64),
                regression=np.zeros((n_cells, 2)),
                constants=np.ones((n_cells, 5)),
                directions=np.zeros((n_cells, 3, 2)),
                method="dr",
                cohorts=np.array([2004.0, 2006.0, 2007.0]),
                biters=99,
                random_state=7,
            ),
        ),
    ]

    restored = [pickle.loads(pickle.dumps(pair, protocol=pickle.HIGHEST_PROTOCOL)) for pair in passes]
    statistics = [
        pickle.loads(pickle.dumps(kernel(shard, state), protocol=pickle.HIGHEST_PROTOCOL)) for kernel, state in restored
    ]

    assert sorted(kernel.__name__ for kernel, _ in passes) == sorted(
        name for name in vars(kernels) if name.endswith("_kernel")
    )
    for (kernel, state), (restored_kernel, restored_state), result in zip(passes, restored, statistics, strict=True):
        assert restored_kernel is kernel
        assert type(restored_state) is type(state)
        np.testing.assert_equal(restored_state._asdict(), state._asdict())
        np.testing.assert_equal(result._asdict(), kernel(shard, state)._asdict())
