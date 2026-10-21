"""Tests for the checks and the unit table that prepare a Spark DataFrame for att_gt."""

import re

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from moderndid import aggte, att_gt, preprocess_did
from moderndid.did.shards.layout import shard_from_arrow
from moderndid.did.shards.local import shards_from_did_data
from tests.helpers import importorskip

importorskip("pyspark")

pytestmark = pytest.mark.spark


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        ("mpdta_data", {"xformla": "~lpopp"}, "'lpopp' in xformla is not a column in the data. Did you mean 'lpop'?"),
        ("mpdta_with_text_outcome", {}, "Validation failed:\nyname = 'lemp' is not numeric. Please convert it"),
        (
            "mpdta_with_repeated_rows",
            {},
            "Validation failed:\nThe value of idname must be unique (by tname). Some units are observed more than "
            "once in a period. Rows repeat for 4 (countyreal, year) pairs, such as (8001, 2006), (8001, 2007), and "
            "(8019, 2006).",
        ),
        (
            "mpdta_with_varying_cohort",
            {},
            "Validation failed:\nThe value of gname (treatment variable) must be the same across all periods for "
            "each particular unit. The treatment must be irreversible.",
        ),
    ],
    ids=["missing_column", "text_outcome", "repeated_rows", "varying_cohort"],
)
def test_att_gt_on_spark_raises_the_errors_of_att_gt(request, to_spark, data_fixture, options, message):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", **options}
    frame = to_spark(data)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(data, **spec)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(frame, **spec)


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        ("mpdta_missing_a_period", {}, "Dropped 45 units while converting to balanced panel"),
        ("mpdta_treated_in_the_first_period", {}, "Dropped 78 units that were already treated in the first period"),
        (
            "mpdta_with_states",
            {"clustervars": ["state"]},
            "The units fall in only 29 clusters. The treated units of cohort 2004 fall in 1 cluster and those of "
            "cohort 2006 in 3 clusters. Cluster-robust inference rests on very few clusters and can give confidence "
            "intervals and bands that are too narrow.",
        ),
    ],
    ids=["units_missing_a_period", "units_treated_in_the_first_period", "few_clusters"],
)
def test_att_gt_on_spark_warns_like_att_gt(request, to_spark, assert_same_fit, data_fixture, options, message):
    data = request.getfixturevalue(data_fixture)
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "random_state": 7,
        **options,
    }
    frame = to_spark(data)

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        memory = att_gt(data, **spec)
    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        distributed = att_gt(frame, **spec)

    assert_same_fit(distributed, memory, 1e-12)


def test_att_gt_on_spark_reads_decimal_periods_and_cohorts_as_doubles_and_decimal_ids_as_integers(
    mpdta_with_decimal_columns, mpdta_with_float_periods, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 99,
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_decimal_columns)

    distributed = att_gt(frame, **spec)
    memory = att_gt(mpdta_with_float_periods, **spec)
    local = att_gt_on_shards(mpdta_with_float_periods, 3, **spec)
    event_study = aggte(distributed, type="dynamic", cband=False)
    expected_event_study = aggte(memory, type="dynamic", cband=False)

    assert [frame.schema[name].dataType.simpleString() for name in ("year", "countyreal", "first.treat")] == [
        "decimal(10,0)"
    ] * 3
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_array_equal(event_study.event_times, expected_event_study.event_times)
    np.testing.assert_allclose(event_study.att_by_event, expected_event_study.att_by_event, rtol=1e-12, atol=1e-12)


def test_att_gt_on_spark_keeps_decimal_ids_above_two_to_the_53_apart_like_integer_ids(
    mpdta_with_large_ids, mpdta_with_large_decimal_ids, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "boot": True,
        "biters": 99,
        "random_state": 7,
    }
    frame = to_spark(mpdta_with_large_decimal_ids)
    ids = np.unique(mpdta_with_large_ids["countyreal"].to_numpy())

    distributed = att_gt(frame, **spec)
    memory = att_gt(mpdta_with_large_ids, **spec)
    local = att_gt_on_shards(mpdta_with_large_ids, 3, **spec)

    assert frame.schema["countyreal"].dataType.simpleString() == "decimal(18,0)"
    assert ids[1] - ids[0] == 1
    assert float(ids[0]) == float(ids[1])
    assert distributed.n_units == 500
    np.testing.assert_allclose(distributed.att_gt, memory.att_gt, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distributed.vcov_analytical, memory.vcov_analytical, rtol=1e-12, atol=1e-12)
    assert_same_fit(distributed, local, 1e-12)


def test_att_gt_on_spark_draws_for_decimal_clusters_as_for_the_same_integer_codes(
    scalable_panel, scalable_panel_with_decimal_clusters, to_spark, att_gt_on_shards, assert_same_fit
):
    spec = {
        "yname": "y",
        "tname": "time",
        "idname": "id",
        "gname": "group",
        "xformla": "~cov1",
        "clustervars": ["cluster"],
        "boot": True,
        "biters": 299,
        "random_state": 3,
    }
    frame = to_spark(scalable_panel_with_decimal_clusters)

    distributed = att_gt(frame, **spec)
    local = att_gt_on_shards(scalable_panel, 3, **spec)
    moments, expected = distributed.influence_moments, local.influence_moments
    analytic = np.sqrt(np.diag(distributed.vcov_analytical) / distributed.n_units)

    assert frame.schema["cluster"].dataType.simpleString() == "decimal(18,0)"
    assert_same_fit(distributed, local, 1e-12)
    np.testing.assert_allclose(moments.cluster_gram, expected.cluster_gram, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(moments.bootstrap_sums, expected.bootstrap_sums, rtol=1e-12, atol=1e-10)
    assert moments.n_clusters == 50
    assert np.nanmax(np.abs(distributed.se_gt / analytic - 1)) > 1e-3


@pytest.mark.parametrize(
    "data_fixture, options, message",
    [
        (
            "mpdta_with_decimal_ids_of_19_digits",
            {},
            "idname='countyreal' is a decimal(19,0) column. Since a Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers, they must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_ids",
            {},
            "idname='countyreal' is a decimal(12,2) column. Since a Spark fit reads decimal identifiers and clusters "
            "as 64-bit integers, they must have no fraction digits and at most 18 digits. Cast the column to bigint or "
            "code the units as integers first.",
        ),
        (
            "mpdta_with_fractional_decimal_states",
            {"clustervars": ["countyreal", "state"]},
            "'state' in clustervars is a decimal(10,2) column. Since a Spark fit reads decimal identifiers and "
            "clusters as 64-bit integers, they must have no fraction digits and at most 18 digits. Cast the column to "
            "bigint or code the clusters as integers first.",
        ),
    ],
    ids=["decimal_ids_of_19_digits", "fractional_decimal_ids", "fractional_decimal_clusters"],
)
def test_att_gt_on_spark_rejects_decimal_ids_and_clusters_that_64_bit_integers_cannot_hold(
    request, to_spark, data_fixture, options, message
):
    frame = to_spark(request.getfixturevalue(data_fixture))

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        att_gt(frame, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **options)


def test_build_unit_table_holds_each_unit_in_the_layout_that_shards_read(
    mpdta_with_weights_and_states, unit_table_of, to_arrow
):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~lpop",
        "weightsname": "w",
        "clustervars": ["state"],
    }

    did_data = preprocess_did(mpdta_with_weights_and_states, panel=True, allow_unbalanced_panel=False, **spec)
    expected = shards_from_did_data(did_data, 1)[0]
    unit_table, config, *_ = unit_table_of(mpdta_with_weights_and_states, **spec)
    shard = shard_from_arrow(to_arrow(unit_table), config.time_periods_count, 1)
    order = np.argsort(shard.unit_keys)
    expected_order = np.argsort(expected.unit_keys)

    assert unit_table.columns == ["unit", "cohort", "y", "w", "x", "cluster"]
    np.testing.assert_array_equal(shard.unit_keys[order], expected.unit_keys[expected_order])
    np.testing.assert_array_equal(shard.cohort[order], expected.cohort[expected_order])
    np.testing.assert_array_equal(shard.y[order], expected.y[expected_order])
    np.testing.assert_allclose(shard.weights[order], expected.weights[expected_order], rtol=1e-14)
    np.testing.assert_array_equal(shard.covariates[order], expected.covariates[expected_order])
    np.testing.assert_array_equal(shard.cluster_keys[order], expected.cluster_keys[expected_order])


@pytest.mark.parametrize(
    "data_fixture, options",
    [
        ("mpdta_missing_a_period", {"xformla": "~lpop"}),
        ("mpdta_without_never_treated", {"control_group": "notyettreated", "anticipation": 1}),
        ("mpdta_with_float_periods", {"xformla": "~lpop"}),
    ],
    ids=["units_missing_a_period", "without_never_treated", "float_periods"],
)
def test_build_unit_table_sets_the_facts_of_the_in_memory_preprocessing(
    request, unit_table_of, record_warnings, data_fixture, options
):
    data = request.getfixturevalue(data_fixture)
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat", **options}

    did_data, memory_warnings = record_warnings(preprocess_did, data, panel=True, allow_unbalanced_panel=False, **spec)
    (_, config, cohort_counts, n_obs, cluster_counts), spark_warnings = record_warnings(unit_table_of, data, **spec)
    expected = did_data.config

    assert spark_warnings == memory_warnings
    np.testing.assert_array_equal(config.time_periods, expected.time_periods)
    np.testing.assert_array_equal(config.treated_groups, expected.treated_groups)
    assert (config.time_periods.dtype, config.treated_groups.dtype) == (
        expected.time_periods.dtype,
        expected.treated_groups.dtype,
    )
    assert (config.time_periods_count, config.treated_groups_count, config.id_count) == (
        expected.time_periods_count,
        expected.treated_groups_count,
        expected.id_count,
    )
    assert (config.cband, config.data_format) == (expected.cband, expected.data_format)
    assert_frame_equal(cohort_counts, did_data.cohort_counts)
    assert n_obs == len(did_data.data)
    assert cluster_counts is None


def test_build_unit_table_counts_the_clusters_of_all_units_and_of_each_cohort(mpdta_with_state_names, unit_table_of):
    spec = {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "clustervars": ["countyreal", "state"],
    }

    units = preprocess_did(mpdta_with_state_names, panel=True, allow_unbalanced_panel=False, **spec).time_invariant_data
    per_cohort = units.group_by("first.treat").agg(pl.col("state").n_unique())
    *_, cluster_counts = unit_table_of(mpdta_with_state_names, **spec)

    assert cluster_counts == (units["state"].n_unique(), dict(per_cohort.iter_rows()))
    assert cluster_counts[0] == 29
