"""Tests for fitting att_gt on a balanced panel held in shards of whole units."""

import pickle
import re
import warnings
from dataclasses import replace
from importlib import import_module
from itertools import pairwise

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning, SingularMatrixWarning

from moderndid import aggte, att_gt, drdid_panel, preprocess_did
from moderndid.core.bootstrap import canonical_keys, keyed_rademacher
from moderndid.did.shards import kernels
from moderndid.did.shards.container import CellSumsState, FitState, MomentsState, Shard, SummaryState
from moderndid.did.shards.driver import fit_att_gt_on_shards
from moderndid.did.shards.kernels import cell_sums_kernel, irls_kernel, moments_kernel, summary_kernel
from moderndid.did.shards.layout import shard_from_arrow
from moderndid.did.shards.local import LocalExecutor, shards_from_did_data
from moderndid.did.shards.plan import plan_cells


def _assert_same_fit(sharded, memory):
    np.testing.assert_array_equal(sharded.groups, memory.groups)
    np.testing.assert_array_equal(sharded.times, memory.times)
    assert (sharded.groups.dtype, sharded.times.dtype) == (memory.groups.dtype, memory.times.dtype)
    assert sharded.n_units == memory.n_units
    np.testing.assert_allclose(sharded.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.se_gt, memory.se_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_value, memory.critical_value, rtol=1e-12)
    wald = [np.nan if result.wald_stat is None else result.wald_stat for result in (sharded, memory)]
    np.testing.assert_allclose(wald[0], wald[1], rtol=1e-12)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("anticipation", [0, 1])
@pytest.mark.parametrize("weightsname", [None, "w"], ids=["unweighted", "time_varying_weights"])
@pytest.mark.parametrize("clustervars", [None, ["countyreal", "state"]], ids=["unclustered", "state_clusters"])
def test_att_gt_on_shards_matches_att_gt_over_the_option_matrix(
    mpdta_varying_weights_and_states,
    att_gt_on_shards,
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

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, **spec)
    shard_messages = [(w.category, str(w.message)) for w in shard_warnings]

    assert [(w.category, str(w.message)) for w in memory_warnings] == [
        (
            UserWarning,
            "anticipation = 1. Never-treated units (with group status 0 or inf) are assumed to never anticipate "
            "treatment. Anticipation only applies to eventually-treated units.",
        )
    ] * anticipation + shard_messages
    assert any(text.startswith("Time-varying weights detected.") for _, text in shard_messages) == bool(weightsname)
    assert any(text.startswith("The units fall in only") for _, text in shard_messages) == bool(clustervars)
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_matches_att_gt_on_a_generated_panel_with_four_covariates(
    scalable_panel, att_gt_on_shards, est_method, control_group, base_period
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

    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_with_clusters_of_a_generated_panel(
    scalable_panel, att_gt_on_shards, est_method
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

    _assert_same_fit(sharded, memory)
    assert sharded.influence_moments.n_clusters == 50


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_without_covariates(mpdta_data, att_gt_on_shards, est_method):
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

    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_matches_att_gt_with_a_covariate_that_varies_over_time(
    mpdta_varying_covariate, att_gt_on_shards, est_method, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~x",
        "est_method": est_method,
        "control_group": "notyettreated",
        "base_period": base_period,
        "random_state": 7,
    }

    memory = att_gt(mpdta_varying_covariate, **spec)
    sharded = att_gt_on_shards(mpdta_varying_covariate, 3, **spec)

    assert mpdta_varying_covariate.group_by("countyreal").agg(pl.col("x").n_unique())["x"].min() == 5
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method, n_extreme", [("dr", 12), ("ipw", 0)])
def test_att_gt_on_shards_matches_att_gt_with_outlying_comparison_units(
    mpdta_outlying_controls, att_gt_on_shards, est_method, n_extreme
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~x",
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(mpdta_outlying_controls, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(mpdta_outlying_controls, 3, **spec)

    assert [str(w.message) for w in memory_warnings] == [
        "Extreme weight ratios detected. Results may be numerically unstable."
    ] * n_extreme
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    _assert_same_fit(sharded, memory)


def test_att_gt_on_shards_treats_clusters_of_one_county_each_as_counties_like_att_gt(
    mpdta_county_clusters, att_gt_on_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "county_cluster"],
        "random_state": 7,
    }

    memory = att_gt(mpdta_county_clusters, **spec)
    sharded = att_gt_on_shards(mpdta_county_clusters, 3, **spec)

    assert sharded.influence_moments.n_clusters == 500
    assert memory.wald_stat > 0
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_matches_att_gt_for_one_or_seven_shards(mpdta_data, att_gt_on_shards, est_method, n_shards):
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

    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_att_gt_with_empty_and_single_unit_shards(mpdta_with_states, est_method):
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
    _assert_same_fit(sharded, memory)


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
def test_att_gt_on_shards_warns_and_drops_cells_like_att_gt(request, att_gt_on_shards, data_fixture, options, message):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "random_state": 7, **options}

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, **spec)

    assert message in [str(w.message) for w in memory_warnings]
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize(
    "options, message",
    [
        ({}, "Overlap condition violated for 2004 in time period 2004"),
        ({"est_method": "ipw", "base_period": "universal"}, "Overlap condition violated for 2004 in time period 2007"),
    ],
)
def test_att_gt_on_shards_warns_like_att_gt_when_a_covariate_separates_a_cohort(
    mpdta_separated_2004, att_gt_on_shards, options, message
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

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(mpdta_separated_2004, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(mpdta_separated_2004, 3, **spec)

    assert message in [str(w.message) for w in memory_warnings]
    assert any(issubclass(w.category, PerfectSeparationWarning) for w in memory_warnings)
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    _assert_same_fit(sharded, memory)


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
    mpdta_separated_2004, att_gt_on_shards, base_period, message
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

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, **spec)

    assert any(str(w.message).startswith(message) for w in memory_warnings)
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method, base_period", [("dr", "varying"), ("reg", "universal")])
def test_att_gt_on_shards_warns_like_att_gt_when_regression_covariates_are_collinear(
    mpdta_collinear_covariates, att_gt_on_shards, est_method, base_period
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
        warnings.simplefilter("ignore", SingularMatrixWarning)
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
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
def test_att_gt_on_shards_with_a_collinear_propensity_covariate_matches_att_gt_without_it(
    mpdta_shifted_covariate, att_gt_on_shards, base_period
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "est_method": "ipw",
        "base_period": base_period,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(mpdta_shifted_covariate, xformla="~lpop", **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(mpdta_shifted_covariate, 3, xformla="~lpop + lpop_shifted", **spec)

    assert [str(w.message) for w in memory_warnings] == []
    assert [str(w.message) for w in shard_warnings] == []
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("data_fixture", ["wide_scalable_panel", "wider_scalable_panel"])
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_att_gt_on_shards_matches_att_gt_without_warnings_on_weighted_panels_with_twenty_and_fifty_covariates(
    request, att_gt_on_shards, data_fixture, est_method, control_group
):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(name for name in data.columns if name.startswith("cov")),
        "weightsname": "w",
        "clustervars": ["cluster"],
        "est_method": est_method,
        "control_group": control_group,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, **spec)

    assert [str(w.message) for w in memory_warnings] == []
    assert [str(w.message) for w in shard_warnings] == []
    np.testing.assert_array_equal(sharded.groups, [2, 2, 3, 3])
    assert np.all(np.isfinite(sharded.att_gt))
    assert np.all(np.isfinite(sharded.se_gt))
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_a_newton_logit_and_least_squares_on_twenty_weighted_covariates(
    wide_scalable_panel, att_gt_on_shards, two_by_two_by_hand, est_method
):
    units = wide_scalable_panel.filter(pl.col("time") == 1).sort("id")
    outcomes = wide_scalable_panel.pivot(on="time", index="id", values="y", sort_columns=True).sort("id")
    covariates = np.column_stack([np.ones(len(units)), units.select(pl.col("^cov\\d+$")).to_numpy()])
    panel = (outcomes.drop("id").to_numpy(), units["group"].to_numpy(), covariates, units["w"].to_numpy())
    cells = [(2, 1, 0, (0,)), (2, 2, 0, (0,)), (3, 1, 0, (0,)), (3, 2, 1, (0,))]
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)),
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
    }

    sharded = att_gt_on_shards(wide_scalable_panel, 3, **spec)
    expected = [two_by_two_by_hand(*panel, cell, est_method) for cell in cells]

    assert covariates.shape == (800, 21)
    np.testing.assert_array_equal(sharded.groups, [2, 2, 3, 3])
    np.testing.assert_array_equal(sharded.times, [2, 3, 2, 3])
    np.testing.assert_allclose(sharded.att_gt, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
def test_att_gt_on_shards_trims_exactly_the_comparison_units_whose_newton_logit_scores_reach_0995(
    trimmed_comparison_panel, att_gt_on_shards, two_by_two_by_hand, est_method
):
    units = trimmed_comparison_panel.filter(pl.col("t") == 1).sort("id")
    outcomes = trimmed_comparison_panel.pivot(on="t", index="id", values="y", sort_columns=True).sort("id")
    covariates = np.column_stack([np.ones(len(units)), units.select(pl.col("^x\\d+$")).to_numpy()])
    panel = (outcomes.drop("id").to_numpy(), units["g"].to_numpy(), covariates, np.ones(len(units)))
    cells = [(3, 1, 0, (0,)), (3, 2, 1, (0,)), (3, 3, 1, (0,))]
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~" + " + ".join(f"x{j}" for j in range(20)),
        "est_method": est_method,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(trimmed_comparison_panel, 3, **spec)
    expected = [two_by_two_by_hand(*panel, cell, est_method) for cell in cells]

    assert [str(w.message) for w in recorded] == []
    np.testing.assert_array_equal(sharded.times, [2, 3, 4])
    np.testing.assert_allclose(sharded.att_gt, expected, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    "redundant",
    [
        pl.col("cov5"),
        0.7 * pl.col("cov5"),
        pl.col("cov5") + 0.5 * pl.col("cov6") - pl.col("cov7"),
        pl.lit(3.0),
    ],
    ids=["copy", "multiple", "combination", "constant"],
)
def test_att_gt_on_shards_ipw_with_a_redundant_twenty_first_covariate_matches_the_fit_without_it(
    wide_scalable_panel, att_gt_on_shards, redundant
):
    data = wide_scalable_panel.with_columns(redundant.alias("redundant"))
    covariates = " + ".join(f"cov{j}" for j in range(1, 21))
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "weightsname": "w",
        "est_method": "ipw",
        "random_state": 7,
    }

    reduced = att_gt_on_shards(data, 3, xformla=f"~{covariates}", **spec)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, xformla=f"~{covariates} + redundant", **spec)

    np.testing.assert_array_equal(reduced.groups, [2, 2, 3, 3])
    assert [str(w.message) for w in recorded] == []
    np.testing.assert_array_equal(sharded.groups, reduced.groups)
    assert np.all(np.isfinite(sharded.att_gt))
    assert np.all(np.isfinite(sharded.se_gt))
    np.testing.assert_allclose(sharded.att_gt, reduced.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(sharded.se_gt, reduced.se_gt, rtol=1e-10, atol=0)


@pytest.mark.parametrize(
    "extra",
    [pl.col("cov5"), pl.when(pl.col("group") == 0).then(0.0).otherwise(pl.col("cov6"))],
    ids=["copy", "zero_among_comparison_units"],
)
@pytest.mark.parametrize("est_method", ["dr", "reg"])
def test_att_gt_on_shards_skips_every_cell_like_att_gt_when_a_wide_comparison_design_is_singular(
    wide_scalable_panel, att_gt_on_shards, extra, est_method
):
    data = wide_scalable_panel.with_columns(extra.alias("extra"))
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)) + " + extra",
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(data, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(data, 3, **spec)

    assert [str(w.message) for w in shard_warnings if w.category is UserWarning] == [
        *(
            f"Not enough control units for group {group} in time period {time} to run specified regression"
            for group, time in ((2, 2), (2, 3), (3, 2), (3, 3))
        ),
        "No pre-treatment periods available for the Wald pre-test of parallel trends. This can happen when all "
        "groups are first treated early in the panel, such as in the second time period, so that no pre-treatment "
        "ATT(g,t) estimates exist.",
    ]
    assert [(w.category, str(w.message)) for w in shard_warnings if w.category is UserWarning] == [
        (w.category, str(w.message)) for w in memory_warnings if w.category is UserWarning
    ]
    assert len(sharded.att_gt) == len(memory.att_gt) == 0


def test_att_gt_on_shards_skips_like_att_gt_the_cells_whose_comparison_group_is_smaller_than_the_coefficients(
    twenty_never_treated_panel, att_gt_on_shards
):
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~" + " + ".join(f"x{j}" for j in range(20)),
        "est_method": "reg",
        "control_group": "notyettreated",
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(twenty_never_treated_panel, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(twenty_never_treated_panel, 3, **spec)

    assert [str(w.message) for w in shard_warnings] == [
        "Be aware that there are some small groups in your dataset.\nCheck groups: inf",
        *(
            f"Not enough control units for group {group} in time period {time} to run specified regression"
            for group, time in ((2, 3), (3, 2), (3, 3))
        ),
        "No pre-treatment periods available for the Wald pre-test of parallel trends. This can happen when all "
        "groups are first treated early in the panel, such as in the second time period, so that no pre-treatment "
        "ATT(g,t) estimates exist.",
    ]
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    np.testing.assert_array_equal(sharded.times, [2])
    assert np.all(np.isfinite(sharded.att_gt))
    assert np.all(np.isfinite(sharded.se_gt))
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_matches_a_newton_logit_and_least_squares_with_four_comparison_units_to_spare(
    twenty_five_never_treated_panel, att_gt_on_shards, two_by_two_by_hand, est_method
):
    units = twenty_five_never_treated_panel.filter(pl.col("t") == 1).sort("id")
    outcomes = twenty_five_never_treated_panel.pivot(on="t", index="id", values="y", sort_columns=True).sort("id")
    covariates = np.column_stack([np.ones(len(units)), units.select(pl.col("^x\\d+$")).to_numpy()])
    panel = (outcomes.drop("id").to_numpy(), units["g"].to_numpy(), covariates, np.ones(len(units)))
    cells = [(2, 1, 0, (0,)), (2, 2, 0, (0,)), (3, 1, 0, (0,)), (3, 2, 1, (0,))]
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~" + " + ".join(f"x{j}" for j in range(20)),
        "est_method": est_method,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(twenty_five_never_treated_panel, 3, **spec)
    expected = [two_by_two_by_hand(*panel, cell, est_method) for cell in cells]

    assert (int(np.sum(panel[1] == 0)), covariates.shape[1]) == (25, 21)
    assert [str(w.message) for w in recorded] == []
    np.testing.assert_array_equal(sharded.times, [2, 3, 2, 3])
    np.testing.assert_allclose(sharded.att_gt, expected, rtol=1e-8, atol=0)


@pytest.mark.parametrize("change", [pl.col("cov5") * 1e4, pl.col("cov5") + 1e3], ids=["rescaled", "shifted"])
@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_keeps_its_estimates_when_one_of_twenty_covariates_is_rescaled_or_shifted(
    wide_scalable_panel, att_gt_on_shards, change, est_method
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)),
        "weightsname": "w",
        "est_method": est_method,
        "random_state": 7,
    }

    original = att_gt_on_shards(wide_scalable_panel, 3, **spec)
    changed = att_gt_on_shards(wide_scalable_panel.with_columns(change.alias("cov5")), 3, **spec)

    np.testing.assert_array_equal(original.groups, [2, 2, 3, 3])
    np.testing.assert_array_equal(changed.groups, original.groups)
    assert np.all(np.isfinite(changed.att_gt))
    assert np.all(np.isfinite(changed.se_gt))
    np.testing.assert_allclose(changed.att_gt, original.att_gt, rtol=1e-9, atol=0)
    np.testing.assert_allclose(changed.se_gt, original.se_gt, rtol=1e-9, atol=0)


@pytest.mark.parametrize("change", [pl.col("cov5") * 1e8, pl.col("cov5") * 1e-8], ids=["times_1e8", "times_1e-8"])
def test_att_gt_on_shards_ipw_keeps_its_estimates_when_one_of_twenty_covariates_is_scaled_by_eight_orders_of_magnitude(
    wide_scalable_panel, att_gt_on_shards, change
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~" + " + ".join(f"cov{j}" for j in range(1, 21)),
        "weightsname": "w",
        "est_method": "ipw",
        "random_state": 7,
    }

    original = att_gt_on_shards(wide_scalable_panel, 3, **spec)
    changed = att_gt_on_shards(wide_scalable_panel.with_columns(change.alias("cov5")), 3, **spec)

    np.testing.assert_array_equal(original.groups, [2, 2, 3, 3])
    np.testing.assert_array_equal(changed.groups, original.groups)
    assert np.all(np.isfinite(changed.att_gt))
    assert np.all(np.isfinite(changed.se_gt))
    np.testing.assert_allclose(changed.att_gt, original.att_gt, rtol=1e-10, atol=0)
    np.testing.assert_allclose(changed.se_gt, original.se_gt, rtol=1e-10, atol=0)


@pytest.mark.parametrize("est_method", ["dr", "ipw", "reg"])
def test_att_gt_on_shards_recovers_the_planted_effects_beside_sixteen_irrelevant_covariates(
    planted_effects_panel, att_gt_on_shards, est_method
):
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~" + " + ".join(f"x{j}" for j in range(20)),
        "est_method": est_method,
        "random_state": 7,
    }

    sharded = att_gt_on_shards(planted_effects_panel, 3, **spec)

    np.testing.assert_array_equal(sharded.groups, [2, 2, 3, 3])
    np.testing.assert_array_equal(sharded.times, [2, 3, 2, 3])
    np.testing.assert_allclose(sharded.att_gt, [1.0, 2.0, 0.0, 1.0], rtol=0, atol=0.5)


@pytest.mark.parametrize("est_method", ["dr", "ipw"])
def test_att_gt_on_shards_drops_like_att_gt_the_cohort_whose_twenty_covariates_push_its_scores_past_0999(
    separated_cohort_panel, att_gt_on_shards, est_method
):
    spec = {
        "yname": "y",
        "tname": "t",
        "idname": "id",
        "gname": "g",
        "xformla": "~" + " + ".join(f"x{j}" for j in range(20)),
        "est_method": est_method,
        "random_state": 7,
    }

    with warnings.catch_warnings(record=True) as memory_warnings:
        warnings.simplefilter("always")
        memory = att_gt(separated_cohort_panel, **spec)
    with warnings.catch_warnings(record=True) as shard_warnings:
        warnings.simplefilter("always")
        sharded = att_gt_on_shards(separated_cohort_panel, 3, **spec)

    assert [str(w.message) for w in shard_warnings] == [
        "Overlap condition violated for 2 in time period 2",
        "Overlap condition violated for 2 in time period 3",
    ]
    assert [(w.category, str(w.message)) for w in shard_warnings] == [
        (w.category, str(w.message)) for w in memory_warnings
    ]
    np.testing.assert_array_equal(sharded.groups, [3, 3])
    assert np.all(np.isfinite(sharded.att_gt))
    assert np.all(np.isfinite(sharded.se_gt))
    _assert_same_fit(sharded, memory)


@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_bootstrap_finishes_keyed_draws_of_the_influence_function(
    monkeypatch, draws_from_weights, mpdta_data, att_gt_on_shards, base_period, n_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": base_period,
        "boot": True,
        "biters": 499,
    }
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]
    signs = keyed_rademacher(canonical_keys(units.to_numpy()), 20261020, 499).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    memory = att_gt(mpdta_data, **spec)
    sharded = att_gt_on_shards(mpdta_data, n_shards, shard_seed=n_shards, random_state=20261020, **spec)

    assert np.isfinite(sharded.se_gt).sum() == 12
    np.testing.assert_allclose(sharded.se_gt, memory.se_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_value, memory.critical_value, rtol=1e-12)


@pytest.mark.parametrize("dtype", [pl.Int64, pl.String])
@pytest.mark.parametrize("n_shards", [1, 7])
def test_att_gt_on_shards_bootstrap_draws_one_keyed_multiplier_for_each_cluster(
    monkeypatch, draws_from_weights, mpdta_with_states, att_gt_on_shards, dtype, n_shards
):
    data = mpdta_with_states.with_columns(pl.col("state").cast(dtype))
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "state"],
        "boot": True,
        "biters": 499,
    }
    units = preprocess_did(data, allow_unbalanced_panel=False, **spec).time_invariant_data
    signs = keyed_rademacher(canonical_keys(np.unique(units["state"].to_numpy())), 20261020, 499).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory = att_gt(data, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded = att_gt_on_shards(data, n_shards, shard_seed=n_shards, random_state=20261020, **spec)

    assert np.isfinite(sharded.se_gt).sum() == 12
    np.testing.assert_allclose(sharded.se_gt, memory.se_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_value, memory.critical_value, rtol=1e-12)


def test_att_gt_on_shards_critical_value_skips_cells_with_a_small_bootstrap_scale(
    monkeypatch, draws_from_weights, did_offsetting_cohort_data, att_gt_on_shards
):
    spec = {"yname": "y", "tname": "t", "idname": "id", "gname": "g", "xformla": "~1", "boot": True, "biters": 999}
    units = preprocess_did(did_offsetting_cohort_data, allow_unbalanced_panel=False, **spec).time_invariant_data["id"]
    signs = keyed_rademacher(canonical_keys(units.to_numpy()), 7, 999).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    memory = att_gt(did_offsetting_cohort_data, **spec)
    sharded = att_gt_on_shards(did_offsetting_cohort_data, 3, random_state=7, **spec)

    assert 2 < memory.critical_value < 7
    np.testing.assert_allclose(sharded.critical_value, memory.critical_value, rtol=1e-12)
    np.testing.assert_allclose(sharded.se_gt, memory.se_gt, rtol=1e-12, atol=1e-12)


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
    np.testing.assert_allclose(moments.gram, rows.T @ rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moments.cluster_gram, cluster_rows.T @ cluster_rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(clusters), 3, 199).T @ cluster_rows,
        rtol=1e-12,
        atol=1e-12,
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
    np.testing.assert_allclose(moments.gram, rows.T @ rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(moments.cluster_gram, cluster_rows.T @ cluster_rows, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(clusters), 3, 99).T @ cluster_rows,
        rtol=1e-12,
        atol=1e-12,
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
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "make, value, other",
    [
        (np.random.SeedSequence, 3, 4),
        (np.random.PCG64, 4, 5),
        (np.random.default_rng, 3, 4),
        (int, 2**64, 2**64 + 1),
        (int, 2**64 + 5, 2**64 + 6),
    ],
    ids=["seed_sequence", "bit_generator", "generator", "two_to_the_64", "above_two_to_the_64"],
)
def test_att_gt_on_shards_derives_its_seed_from_a_seed_sequence_a_bit_generator_a_generator_or_a_large_integer(
    mpdta_data, att_gt_on_shards, make, value, other
):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"}
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]

    memory = att_gt(mpdta_data, cband=False, **spec)
    fit = att_gt_on_shards(mpdta_data, 3, boot=True, biters=99, random_state=make(value), **spec)
    again = att_gt_on_shards(mpdta_data, 7, shard_seed=1, boot=True, biters=99, random_state=make(value), **spec)
    another = att_gt_on_shards(mpdta_data, 3, boot=True, biters=99, random_state=make(other), **spec)
    event_study = aggte(fit, type="dynamic")
    seed = fit.influence_moments.random_state
    cohorts = np.asarray(memory.G, dtype=float)
    rows = np.column_stack(
        [memory.influence_func, *(cohorts == g for g in (2004, 2006, 2007)), np.ones(memory.n_units)]
    )

    assert isinstance(seed, int)
    assert 0 <= seed < 2**64
    assert (fit.estimation_params["random_state"], event_study.estimation_params["random_state"]) == (seed, seed)
    assert again.influence_moments.random_state == seed
    assert another.influence_moments.random_state != seed
    np.testing.assert_allclose(
        fit.influence_moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(units.to_numpy()), seed, 99).T @ rows,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(again.se_gt, fit.se_gt, rtol=1e-12)


def test_att_gt_on_shards_draws_from_an_integer_seed_below_2_to_the_64_as_given(mpdta_data, att_gt_on_shards):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", "xformla": "~lpop"}
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]

    memory = att_gt(mpdta_data, cband=False, **spec)
    fit = att_gt_on_shards(mpdta_data, 3, boot=True, biters=99, random_state=2**64 - 1, **spec)
    cohorts = np.asarray(memory.G, dtype=float)
    rows = np.column_stack(
        [memory.influence_func, *(cohorts == g for g in (2004, 2006, 2007)), np.ones(memory.n_units)]
    )

    assert (fit.influence_moments.random_state, fit.estimation_params["random_state"]) == (2**64 - 1, 2**64 - 1)
    np.testing.assert_allclose(
        fit.influence_moments.bootstrap_sums,
        keyed_rademacher(canonical_keys(units.to_numpy()), 2**64 - 1, 99).T @ rows,
        rtol=1e-12,
        atol=1e-12,
    )


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
def test_aggte_bootstrap_on_shards_finishes_keyed_draws_of_the_event_study(
    monkeypatch, draws_from_weights, mpdta_data, att_gt_on_shards, n_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "base_period": "universal",
    }
    units = preprocess_did(mpdta_data, allow_unbalanced_panel=False, **spec).time_invariant_data["countyreal"]
    signs = keyed_rademacher(canonical_keys(units.to_numpy()), 20261020, 250).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    memory = aggte(att_gt(mpdta_data, cband=False, **spec), type="dynamic", boot=True, biters=250, cband=True)
    sharded_fit = att_gt_on_shards(
        mpdta_data, n_shards, shard_seed=n_shards, boot=True, biters=499, random_state=20261020, **spec
    )
    sharded = aggte(sharded_fit, type="dynamic", biters=250)

    assert np.isfinite(sharded.se_by_event).sum() == 7
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, memory.critical_values, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


def test_aggte_on_a_clustered_shard_fit_draws_its_band_from_the_stored_cluster_draws(
    monkeypatch, draws_from_weights, mpdta_with_states, att_gt_on_shards
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
    signs = keyed_rademacher(canonical_keys(np.unique(units["state"].to_numpy())), 11, 399).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory_fit = att_gt(mpdta_with_states, cband=False, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded_fit = att_gt_on_shards(mpdta_with_states, 7, biters=399, random_state=11, **spec)
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        memory = aggte(memory_fit, type="dynamic", biters=399, cband=True)
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        sharded = aggte(sharded_fit, type="dynamic")

    assert 2 < memory.critical_values[0] < 7
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, memory.critical_values, rtol=1e-12)


def test_aggte_bootstrap_on_a_clustered_shard_fit_finishes_the_stored_cluster_draws(
    monkeypatch, draws_from_weights, mpdta_with_states, att_gt_on_shards
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
    signs = keyed_rademacher(canonical_keys(np.unique(units["state"].to_numpy())), 11, 399).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        memory_fit = att_gt(mpdta_with_states, cband=False, **spec)
    with pytest.warns(UserWarning, match="^The units fall in only 29 clusters"):
        sharded_fit = att_gt_on_shards(mpdta_with_states, 7, boot=True, biters=399, random_state=11, **spec)
    memory = aggte(memory_fit, type="dynamic", boot=True, biters=399, cband=True)
    sharded = aggte(sharded_fit, type="dynamic")

    assert np.isfinite(sharded.se_by_event).sum() == len(sharded.event_times)
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, memory.critical_values, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


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
            "clustervars=[] asks for unit-level bootstrap draws. Since this att_gt fit on a Spark DataFrame drew "
            "one multiplier for each cluster, it holds no draws for single units."
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


@pytest.mark.parametrize("aggregation, cband", [("dynamic", False), ("simple", True)])
def test_aggte_on_shards_without_stored_draws_gives_analytic_results_when_it_needs_no_draws(
    mpdta_data, att_gt_on_shards, aggregation, cband
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "cband": False,
    }

    memory = aggte(att_gt(mpdta_data, **spec), type=aggregation, cband=cband)
    sharded_fit = att_gt_on_shards(mpdta_data, 3, **spec)
    sharded = aggte(sharded_fit, type=aggregation, cband=cband)

    assert sharded_fit.influence_moments.bootstrap_sums is None
    np.testing.assert_allclose(sharded.overall_att, memory.overall_att, rtol=1e-12)
    np.testing.assert_allclose(sharded.overall_se, memory.overall_se, rtol=1e-12)


def test_aggte_on_shards_draws_by_county_from_a_fit_whose_clusters_hold_one_county_each(
    monkeypatch, draws_from_weights, mpdta_county_clusters, att_gt_on_shards
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "clustervars": ["countyreal", "county_cluster"],
    }
    units = preprocess_did(mpdta_county_clusters, allow_unbalanced_panel=False, **spec).time_invariant_data
    signs = keyed_rademacher(canonical_keys(units["county_cluster"].to_numpy()), 7, 1000).T
    monkeypatch.setattr(import_module("moderndid.did.mboot"), "_run_multiplier_bootstrap", draws_from_weights(signs))

    memory_fit = att_gt(mpdta_county_clusters, cband=False, **spec)
    sharded_fit = att_gt_on_shards(mpdta_county_clusters, 3, random_state=7, **spec)
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        memory = aggte(memory_fit, type="dynamic", clustervars=[], biters=1000, cband=True)
    with pytest.warns(UserWarning, match="^Used bootstrap procedure to compute simultaneous confidence band$"):
        sharded = aggte(sharded_fit, type="dynamic", clustervars=[])

    assert 2 < memory.critical_values[0] < 7
    np.testing.assert_allclose(sharded.se_by_event, memory.se_by_event, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sharded.critical_values, memory.critical_values, rtol=1e-12)


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
            "The shard engine fits balanced panels only. Pass panel=True and allow_unbalanced_panel=False.",
        ),
        (
            {"allow_unbalanced_panel": True},
            "The shard engine fits balanced panels only. Pass panel=True and allow_unbalanced_panel=False.",
        ),
        (
            {"est_method": "probit"},
            "est_method='probit' is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'.",
        ),
        ({"est_method": drdid_panel}, " is not valid for a fit on shards. Must be 'dr', 'ipw', or 'reg'."),
    ],
    ids=["repeated_cross_sections", "unbalanced_panel", "unknown_estimator", "custom_estimator"],
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


def test_fit_att_gt_on_shards_rejects_shards_with_a_cohort_that_the_cohort_counts_leave_out(mpdta_data):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=False
    )
    executor = LocalExecutor(shards_from_did_data(did_data, 3, random_state=0))

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "A shard holds a unit whose cohort the panel's cohort counts leave out. Rebuild the shards from the "
            "panel that set the cohort counts."
        ),
    ):
        fit_att_gt_on_shards(
            executor,
            did_data.config,
            did_data.cohort_counts.filter(pl.col("cohort") != 2006),
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


@pytest.mark.parametrize("n_shards", [1, np.int64(3), 7])
def test_shards_from_did_data_puts_each_unit_whole_into_one_shard(mpdta_varying_covariate, n_shards):
    did_data = preprocess_did(
        mpdta_varying_covariate,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~x",
        clustervars=["countyreal", "state"],
        allow_unbalanced_panel=False,
    )
    outcomes = mpdta_varying_covariate.pivot(on="year", index="countyreal", values="lemp", sort_columns=True)
    covariates = mpdta_varying_covariate.pivot(on="year", index="countyreal", values="x", sort_columns=True)
    counties = mpdta_varying_covariate.filter(pl.col("year") == 2003).sort("countyreal")

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


def test_shards_from_did_data_leaves_out_covariates_and_clusters_the_fit_does_not_use(mpdta_data):
    did_data = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~1",
        clustervars=["countyreal"],
        allow_unbalanced_panel=False,
    )

    shards = shards_from_did_data(did_data, 3, random_state=5)

    assert [(shard.weights is None, shard.covariates is None, shard.cluster_keys is None) for shard in shards] == [
        (True, True, True)
    ] * 3


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
            "shards_from_did_data needs a balanced panel. Preprocess the data with panel=True and "
            "allow_unbalanced_panel=False."
        ),
    ):
        shards_from_did_data(did_data, 3)


def test_shards_from_did_data_rejects_a_panel_preprocessed_as_unbalanced(mpdta_data):
    did_data = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", allow_unbalanced_panel=True
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "shards_from_did_data needs a balanced panel. Preprocess the data with panel=True and "
            "allow_unbalanced_panel=False."
        ),
    ):
        shards_from_did_data(did_data, 3)


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
    np.testing.assert_equal(from_table._asdict(), from_batch._asdict())
    np.testing.assert_array_equal(from_table.covariates[:, :, 0], [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])


@pytest.mark.parametrize(
    "outcomes",
    [[[1.0, 2.0], [3.0]], [[1.0, 2.0], None], [[1.0, 2.0, 3.0, 4.0], None]],
    ids=["short_list", "null_list", "null_list_beside_a_long_list"],
)
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
    assert [kernel for kernel, _ in restored] == [kernel for kernel, _ in passes]
    assert [type(state) for _, state in restored] == [type(state) for _, state in passes]
    np.testing.assert_equal([state._asdict() for _, state in restored], [state._asdict() for _, state in passes])
    np.testing.assert_equal(
        [result._asdict() for result in statistics], [kernel(shard, state)._asdict() for kernel, state in passes]
    )
