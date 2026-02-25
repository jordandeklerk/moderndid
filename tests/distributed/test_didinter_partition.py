"""Tests for partition-level array building and estimation in DIDInter."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.distributed._didinter_partition import (
    apply_trends_lin_accumulation,
    build_didinter_partition_arrays,
    partition_apply_control_adjustment,
    partition_apply_globals,
    partition_check_group_level_covariates,
    partition_compute_influence,
    partition_compute_variance_part2,
    partition_control_gram,
    partition_control_influence_sums,
    partition_count_obs,
    partition_delta_and_variance,
    partition_delta_d,
    partition_dof_stats,
    partition_extract_het_data,
    partition_global_scalars,
    partition_group_sums,
    partition_group_sums_and_scalars,
    partition_horizon_covariate_ops,
    partition_horizon_local_ops,
    partition_influence_and_meta,
    partition_variance_influence,
    prepare_het_sample,
    reduce_control_gram,
    reduce_control_influence_sums,
    reduce_dof_stats,
    reduce_global_scalars,
    reduce_group_sums,
    solve_control_coefficients,
)


def _run_horizon(part, abs_h=1, horizon_type="effect", config=None, t_max=5.0):
    if config is None:
        config = {"switchers": ""}
    return partition_horizon_local_ops(part, abs_h=abs_h, horizon_type=horizon_type, config_dict=config, t_max=t_max)


def _run_through_globals(part, abs_h=1, t_max=5.0):
    part = _run_horizon(part, abs_h=abs_h, t_max=t_max)
    gs = partition_group_sums(part, abs_h=abs_h)
    sc = partition_global_scalars(part, abs_h=abs_h)
    part = partition_apply_globals(part, abs_h=abs_h, global_group_sums=gs)
    return part, gs, sc


class TestBuildPartitionArrays:
    def test_polars_input_produces_float64_arrays(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "d": [0, 0, 1, 1],
                "F_g": [float("inf"), float("inf"), 2.0, 2.0],
                "L_g": [float("inf"), float("inf"), 1.0, 1.0],
                "S_g": [0, 0, 1, 1],
                "d_sq": [0.0, 0.0, 0.0, 0.0],
                "weight_gt": [1.0, 1.0, 1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0, 1.0, 0.0],
                "T_g": [2.0, 2.0, 2.0, 2.0],
            }
        )
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d"}
        result = build_didinter_partition_arrays(pdf, col_config)

        assert result["y"].dtype == np.float64
        assert result["gname"].dtype == np.float64
        np.testing.assert_array_equal(result["y"], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result["gname"], [1.0, 1.0, 2.0, 2.0])
        assert result["n_rows"] == 4

    def test_pandas_input_produces_identical_result(self):
        pd = importorskip("pandas")
        data = {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "d": [0, 0, 1, 1],
            "F_g": [float("inf"), float("inf"), 2.0, 2.0],
            "L_g": [float("inf"), float("inf"), 1.0, 1.0],
            "S_g": [0, 0, 1, 1],
            "d_sq": [0.0, 0.0, 0.0, 0.0],
            "weight_gt": [1.0, 1.0, 1.0, 1.0],
            "first_obs_by_gp": [1.0, 0.0, 1.0, 0.0],
            "T_g": [2.0, 2.0, 2.0, 2.0],
        }
        pdf_pl = pl.DataFrame(data)
        pdf_pd = pd.DataFrame(data)
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d"}

        result_pl = build_didinter_partition_arrays(pdf_pl, col_config)
        result_pd = build_didinter_partition_arrays(pdf_pd, col_config)

        for key in ("gname", "tname", "y", "d", "F_g", "S_g"):
            np.testing.assert_array_equal(result_pl[key], result_pd[key])

    @pytest.mark.parametrize("optional_col", ["d_fg", "same_switcher_valid", "t_max_by_group"])
    def test_optional_columns_included_when_present(self, optional_col):
        pdf = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                optional_col: [1.0, 1.0],
            }
        )
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d"}
        result = build_didinter_partition_arrays(pdf, col_config)
        assert optional_col in result

    def test_cluster_column_extracted_as_raw(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                "cluster_id": ["A", "A"],
            }
        )
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d", "cluster": "cluster_id"}
        result = build_didinter_partition_arrays(pdf, col_config)
        np.testing.assert_array_equal(result["cluster"], ["A", "A"])

    @pytest.mark.parametrize("config_key,col_name", [("covariate_names", "x1"), ("trends_nonparam", "trend_var")])
    def test_covariates_and_trends_extracted(self, config_key, col_name):
        pdf = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                col_name: [0.5, 0.6],
            }
        )
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d", config_key: [col_name]}
        result = build_didinter_partition_arrays(pdf, col_config)
        np.testing.assert_array_equal(result[col_name], [0.5, 0.6])

    def test_het_covariates_not_duplicated_when_already_in_covariates(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                "x1": [0.5, 0.6],
            }
        )
        col_config = {
            "gname": "id",
            "tname": "time",
            "yname": "y",
            "dname": "d",
            "covariate_names": ["x1"],
            "het_covariates": ["x1"],
        }
        result = build_didinter_partition_arrays(pdf, col_config)
        np.testing.assert_array_equal(result["x1"], [0.5, 0.6])


class TestHorizonLocalOps:
    @pytest.mark.parametrize(
        "unit_g,expected_diffs",
        [
            (1.0, [1.0, 3.0, 1.0, 1.0]),
            (2.0, [1.0, 1.0, 3.5, 1.0]),
            (3.0, [0.0, 0.0, 0.0, 0.0]),
        ],
    )
    def test_effect_diff_y_computes_first_difference(self, basic_partition, unit_g, expected_diffs):
        part = _run_horizon(basic_partition)
        diff_y = part["diff_y_1"]
        gname = part["gname"]
        unit_diff = diff_y[gname == unit_g]
        np.testing.assert_allclose(unit_diff[1:], expected_diffs)

    def test_anticipation_uses_double_lag_difference(self, basic_partition):
        part = _run_horizon(basic_partition, horizon_type="anticipation")
        diff_y = part["diff_y_1"]
        gname = part["gname"]
        y = part["y"]

        for g in [1.0, 2.0, 3.0]:
            mask = gname == g
            unit_y = y[mask]
            unit_diff = diff_y[mask]
            np.testing.assert_allclose(unit_diff[2], unit_y[0] - unit_y[1])

    def test_never_change_one_for_never_switcher_and_zero_post_switch(self, basic_partition):
        part = _run_horizon(basic_partition)
        never_change = part["never_change_1"]
        gname = part["gname"]
        tname = part["tname"]

        unit3_valid = never_change[(gname == 3) & ~np.isnan(never_change)]
        np.testing.assert_array_equal(unit3_valid, 1.0)

        unit1_post = never_change[(gname == 1) & (tname >= 3) & ~np.isnan(never_change)]
        np.testing.assert_array_equal(unit1_post, 0.0)

    def test_only_never_switchers_removes_partial_controls(self, basic_partition):
        config = {"only_never_switchers": True, "switchers": ""}
        part = _run_horizon(basic_partition, config=config)
        never_change = part["never_change_1"]
        F_g = part["F_g"]
        tname = part["tname"]
        partial_mask = ~np.isnan(never_change) & (F_g > tname) & (F_g < 6.0)
        np.testing.assert_array_equal(never_change[partial_mask], 0.0)

    @pytest.mark.parametrize(
        "direction,expected_in,expected_out",
        [
            ("in", True, False),
            ("out", False, True),
        ],
    )
    def test_switcher_mask_filters_by_direction(self, bidirectional_partition, direction, expected_in, expected_out):
        config = {"switchers": direction}
        part = _run_horizon(bidirectional_partition, config=config)
        sw_mask = part["switcher_mask_1"]
        S_g = part["S_g"]
        assert np.all(sw_mask[S_g == 1]) == expected_in
        assert np.all(sw_mask[S_g == -1]) == expected_out

    def test_dist_raw_marks_correct_treated_periods(self, basic_partition):
        part = _run_horizon(basic_partition)
        dist_raw = part["dist_raw_1"]
        gname = part["gname"]
        tname = part["tname"]

        for g, fg in [(1.0, 3.0), (2.0, 4.0)]:
            val = dist_raw[(gname == g) & (tname == fg)]
            assert val[0] == 1.0

    def test_weighted_diff_scales_by_weight(self, basic_partition):
        basic_partition["weight_gt"] = np.full(basic_partition["n_rows"], 2.0)
        part = _run_horizon(basic_partition)
        expected = np.nan_to_num(part["diff_y_1"], nan=0.0) * 2.0
        np.testing.assert_allclose(part["weighted_diff_1"], expected)

    def test_same_switchers_excludes_invalid_units(self, basic_partition):
        basic_partition["same_switcher_valid"] = np.where(basic_partition["gname"] == 1, 1.0, 0.0)
        config = {"switchers": "", "same_switchers": True}
        part = _run_horizon(basic_partition, config=config)
        assert not np.any(part["switcher_mask_1"][part["gname"] == 2])

    def test_horizon_too_large_leaves_all_nan(self, basic_partition):
        n = basic_partition["n_rows"]
        part = _run_horizon(basic_partition, abs_h=n + 1)
        assert np.all(np.isnan(part[f"diff_y_{n + 1}"]))


class TestGroupSums:
    def test_keys_match_unique_time_dose_pairs(self, basic_partition):
        part = _run_horizon(basic_partition)
        gs = partition_group_sums(part, abs_h=1)
        expected_keys = set(map(tuple, np.unique(np.column_stack([part["tname"], part["d_sq"]]), axis=0)))
        assert set(gs.keys()) == expected_keys

    def test_controls_counted_from_never_switcher(self, basic_partition):
        part = _run_horizon(basic_partition)
        gs = partition_group_sums(part, abs_h=1)
        total_control = sum(v["n_control"] for v in gs.values())
        assert total_control > 0

    def test_reduce_adds_counts_and_preserves_disjoint_keys(self):
        a = {(1.0, 0.0): {"n_control": 5.0, "n_treated": 2.0}}
        b = {
            (1.0, 0.0): {"n_control": 3.0, "n_treated": 1.0},
            (2.0, 0.0): {"n_control": 4.0, "n_treated": 0.0},
        }
        merged = reduce_group_sums(a, b)
        assert merged[(1.0, 0.0)]["n_control"] == 8.0
        assert merged[(1.0, 0.0)]["n_treated"] == 3.0
        assert merged[(2.0, 0.0)]["n_control"] == 4.0


class TestGlobalScalars:
    def test_switcher_gnames_identified(self, basic_partition):
        part = _run_horizon(basic_partition)
        sc = partition_global_scalars(part, abs_h=1)
        assert sc["switcher_gnames"] == {1.0, 2.0}
        assert sc["n_switchers_weighted"] > 0

    def test_reduce_unions_gnames_and_adds_weights(self):
        a = {"n_switchers_weighted": 10.0, "switcher_gnames": {1.0, 2.0}}
        b = {"n_switchers_weighted": 5.0, "switcher_gnames": {2.0, 3.0}}
        merged = reduce_global_scalars(a, b)
        assert merged["n_switchers_weighted"] == 15.0
        assert merged["switcher_gnames"] == {1.0, 2.0, 3.0}


class TestApplyGlobals:
    def test_n_control_constant_within_time_dose_cell(self, basic_partition):
        part, _gs, _ = _run_through_globals(basic_partition)
        n_control = part["n_control_1"]
        tname = part["tname"]
        d_sq = part["d_sq"]

        for t in np.unique(tname):
            for d in np.unique(d_sq):
                vals = n_control[(tname == t) & (d_sq == d)]
                if len(vals) > 0:
                    assert np.all(vals == vals[0])

    def test_dist_zeroed_when_no_controls(self, basic_partition):
        part = _run_horizon(basic_partition)
        tname = part["tname"]
        d_sq = part["d_sq"]
        stacked = np.column_stack([tname, d_sq])
        fake_gs = {tuple(row): {"n_control": 0.0, "n_treated": 1.0} for row in np.unique(stacked, axis=0)}

        part = partition_apply_globals(part, abs_h=1, global_group_sums=fake_gs)
        valid = ~np.isnan(part["dist_1"])
        np.testing.assert_array_equal(part["dist_1"][valid], 0.0)


class TestComputeInfluence:
    def test_influence_values_sum_to_partial_sum(self, basic_partition):
        part, _, sc = _run_through_globals(basic_partition)
        gname_if, partial_sum = partition_compute_influence(
            part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"]
        )
        np.testing.assert_allclose(sum(gname_if.values()), partial_sum)

    def test_influence_zero_at_non_first_obs_rows(self, basic_partition):
        part, _, sc = _run_through_globals(basic_partition)
        partition_compute_influence(part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"])
        np.testing.assert_array_equal(part["inf_col_1"][part["first_obs_by_gp"] == 0], 0.0)

    def test_influence_keys_are_all_unit_ids(self, basic_partition):
        part, _, sc = _run_through_globals(basic_partition)
        gname_if, _ = partition_compute_influence(
            part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"]
        )
        assert set(gname_if.keys()) == {1.0, 2.0, 3.0}

    def test_influence_scales_linearly_with_n_groups(self, large_partition):
        part, _, sc = _run_through_globals(large_partition, t_max=8.0)
        n_sw = sc["n_switchers_weighted"]
        _, sum_5 = partition_compute_influence(part, abs_h=1, n_groups=5, n_switchers_weighted=n_sw)
        _, sum_10 = partition_compute_influence(part, abs_h=1, n_groups=10, n_switchers_weighted=n_sw)
        np.testing.assert_allclose(sum_10 / sum_5, 2.0, rtol=1e-10)


class TestDeltaD:
    @pytest.mark.parametrize(
        "direction,unit,expected_positive",
        [
            ("", 1.0, True),
            ("", 2.0, True),
        ],
    )
    def test_dose_change_sign(self, basic_partition, direction, unit, expected_positive):
        part, _, _sc = _run_through_globals(basic_partition)
        dist_col = part.get("dist_1")
        gname = part["gname"]
        weight = part["weight_gt"]
        valid = (~np.isnan(dist_col)) & (dist_col == 1.0) & (gname == unit)
        sw_weights = dict(zip(gname[valid].tolist(), weight[valid].tolist(), strict=False))
        contrib, _ = partition_delta_d(part, abs_h=1, _horizon_type="effect", switcher_gnames_with_weight=sw_weights)
        assert (contrib > 0) == expected_positive

    def test_empty_switcher_map_gives_zero(self, basic_partition):
        part = _run_horizon(basic_partition)
        contrib, w = partition_delta_d(part, abs_h=1, _horizon_type="effect", switcher_gnames_with_weight={})
        assert contrib == 0.0
        assert w == 0.0

    def test_out_switcher_positive_contrib(self, bidirectional_partition):
        part, _, _ = _run_through_globals(bidirectional_partition)
        contrib, _ = partition_delta_d(part, abs_h=1, _horizon_type="effect", switcher_gnames_with_weight={2.0: 1.0})
        assert contrib > 0


class TestDofStats:
    @pytest.mark.parametrize("section", ["switcher", "control", "union"])
    def test_dof_stats_has_all_sections(self, basic_partition, section):
        part, _, _ = _run_through_globals(basic_partition)
        dof = partition_dof_stats(part, abs_h=1)
        assert section in dof

    def test_switcher_and_control_sections_nonempty(self, basic_partition):
        part, _, _ = _run_through_globals(basic_partition)
        dof = partition_dof_stats(part, abs_h=1)
        assert len(dof["switcher"]) > 0
        assert len(dof["control"]) > 0

    def test_cluster_sets_populated(self, partition_with_clusters):
        part, _, _ = _run_through_globals(partition_with_clusters)
        dof = partition_dof_stats(part, abs_h=1, cluster_col="cluster")
        for section in ("switcher", "control", "union"):
            for info in dof[section].values():
                if info["count"] > 0:
                    assert len(info["cluster_set"]) > 0

    def test_reduce_merges_counts_and_cluster_sets(self):
        a = {
            "switcher": {(0.0, 3.0, 0.0): {"weight_sum": 1.0, "diff_sum": 0.5, "count": 3, "cluster_set": {"A"}}},
            "control": {},
            "union": {},
        }
        b = {
            "switcher": {(0.0, 3.0, 0.0): {"weight_sum": 2.0, "diff_sum": 1.0, "count": 5, "cluster_set": {"B"}}},
            "control": {(2.0, 0.0): {"weight_sum": 1.0, "diff_sum": 0.3, "count": 2, "cluster_set": set()}},
            "union": {},
        }
        merged = reduce_dof_stats(a, b)
        sw = merged["switcher"][(0.0, 3.0, 0.0)]
        assert sw["weight_sum"] == 3.0
        assert sw["diff_sum"] == 1.5
        assert sw["count"] == 8
        assert sw["cluster_set"] == {"A", "B"}
        assert len(merged["control"]) == 1


class TestVarianceInfluence:
    def test_returns_per_unit_dict(self, basic_partition):
        part, _, sc = _run_through_globals(basic_partition)
        dof = partition_dof_stats(part, abs_h=1)
        var_if = partition_variance_influence(
            part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"], global_dof=dof
        )
        assert isinstance(var_if, dict)
        assert len(var_if) > 0

    def test_less_conservative_se_differs_from_default(self, large_partition):
        part, _, sc = _run_through_globals(large_partition, t_max=8.0)
        partition_compute_influence(part, abs_h=1, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"])
        dof = partition_dof_stats(part, abs_h=1)
        n_sw = sc["n_switchers_weighted"]

        var_cons = partition_variance_influence(part, 1, 20, n_sw, dof, less_conservative_se=False)
        var_less = partition_variance_influence(part, 1, 20, n_sw, dof, less_conservative_se=True)

        cons_vals = np.array([var_cons.get(float(g), 0.0) for g in range(1, 21)])
        less_vals = np.array([var_less.get(float(g), 0.0) for g in range(1, 21)])
        assert not np.allclose(cons_vals, less_vals)


def test_count_obs_positive_with_influence(basic_partition):
    part, _, sc = _run_through_globals(basic_partition)
    partition_compute_influence(part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"])
    count = partition_count_obs(part, abs_h=1)
    assert count > 0
    assert count <= basic_partition["n_rows"]


def test_count_obs_zero_without_inf_temp(basic_partition):
    part = _run_horizon(basic_partition)
    count = partition_count_obs(part, abs_h=1)
    assert count == 0


class TestHorizonCovariateOps:
    def test_lag_is_previous_value_within_unit(self, partition_with_covariates):
        part = _run_horizon(partition_with_covariates)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["x1"])
        lag = part["lag_x1_1"]
        x1 = part["x1"]
        gname = part["gname"]

        for g in np.unique(gname):
            mask = gname == g
            assert np.isnan(lag[mask][0])
            np.testing.assert_allclose(lag[mask][1:], x1[mask][:-1])

    def test_diff_equals_value_minus_lag(self, partition_with_covariates):
        part = _run_horizon(partition_with_covariates)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["x1"])
        valid = ~np.isnan(part["lag_x1_1"])
        np.testing.assert_allclose(part["diff_x1_1"][valid], part["x1"][valid] - part["lag_x1_1"][valid])

    def test_missing_covariate_produces_all_nan(self, basic_partition):
        part = _run_horizon(basic_partition)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["nonexistent"])
        assert np.all(np.isnan(part["lag_nonexistent_1"]))
        assert np.all(np.isnan(part["diff_nonexistent_1"]))


class TestControlGramAndSolve:
    def _make_control_partition(self, rng):
        n_periods = 6
        n = 4 * n_periods
        gname = np.repeat([1.0, 2.0, 3.0, 4.0], n_periods)
        tname = np.tile(np.arange(1.0, n_periods + 1), 4)
        F_g = np.where(gname <= 2, 4.0, np.inf)
        S_g = np.where(gname <= 2, 1.0, 0.0)
        d_sq = np.zeros(n)
        L_g = np.where(gname <= 2, 3.0, np.inf)
        T_g = np.full(n, float(n_periods))
        x1 = rng.standard_normal(n)
        y = rng.standard_normal(n) * 0.1 + 1.0 * x1
        weight_gt = np.ones(n)
        d = np.zeros(n)
        d[(gname <= 2) & (tname >= 4)] = 1.0
        first_obs = np.zeros(n)
        for g in [1.0, 2.0, 3.0, 4.0]:
            first_obs[np.where(gname == g)[0][0]] = 1.0
        return {
            "gname": gname,
            "tname": tname,
            "y": y,
            "d": d,
            "F_g": F_g,
            "L_g": L_g,
            "S_g": S_g,
            "d_sq": d_sq,
            "weight_gt": weight_gt,
            "first_obs_by_gp": first_obs,
            "T_g": T_g,
            "n_rows": n,
            "x1": x1,
        }

    def test_gram_computed_for_never_switchers_only(self, rng):
        part = self._make_control_partition(rng)
        part = _run_horizon(part, t_max=6.0)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["x1"])
        gram = partition_control_gram(part, abs_h=1, covariate_names=["x1"])
        assert 0.0 in gram
        assert gram[0.0]["group_set"].issubset({3.0, 4.0})

    def test_reduce_sums_matrices_and_unions_groups(self):
        a = {0.0: {"XtWX": np.array([[2.0]]), "XtWy": np.array([1.0]), "group_set": {1.0}, "weight_sum": 5.0}}
        b = {0.0: {"XtWX": np.array([[3.0]]), "XtWy": np.array([2.0]), "group_set": {2.0}, "weight_sum": 3.0}}
        merged = reduce_control_gram(a, b)
        np.testing.assert_array_equal(merged[0.0]["XtWX"], np.array([[5.0]]))
        np.testing.assert_array_equal(merged[0.0]["XtWy"], np.array([3.0]))
        assert merged[0.0]["group_set"] == {1.0, 2.0}
        assert merged[0.0]["weight_sum"] == 8.0

    def test_reduce_handles_disjoint_levels(self):
        a = {0.0: {"XtWX": np.eye(1), "XtWy": np.ones(1), "group_set": {1.0}, "weight_sum": 1.0}}
        b = {1.0: {"XtWX": np.eye(1), "XtWy": np.ones(1), "group_set": {2.0}, "weight_sum": 1.0}}
        merged = reduce_control_gram(a, b)
        assert 0.0 in merged and 1.0 in merged

    @pytest.mark.parametrize(
        "det_value,expected_useful",
        [
            (0.0, False),
            (None, False),
        ],
    )
    def test_solve_with_degenerate_gram(self, det_value, expected_useful):
        if det_value == 0.0:
            gram = {0.0: {"XtWX": np.zeros((2, 2)), "XtWy": np.zeros(2), "group_set": {1.0, 2.0}, "weight_sum": 10.0}}
        else:
            gram = {0.0: {"XtWX": np.eye(2), "XtWy": np.ones(2), "group_set": {1.0}, "weight_sum": 1.0}}
        result = solve_control_coefficients(gram, n_controls=2, n_groups=5)
        assert not result[0.0]["useful"]

    def test_solve_recovers_true_coefficient(self):
        rng = np.random.default_rng(99)
        n_obs = 200
        x = rng.standard_normal(n_obs)
        y = 2.0 * x + rng.standard_normal(n_obs) * 0.01
        X = x.reshape(-1, 1)
        w = np.ones(n_obs)
        gram = {
            0.0: {
                "XtWX": (X * w[:, None]).T @ X,
                "XtWy": (X * w[:, None]).T @ y,
                "group_set": set(range(n_obs)),
                "weight_sum": float(n_obs),
            }
        }
        result = solve_control_coefficients(gram, n_controls=1, n_groups=n_obs)
        assert result[0.0]["useful"]
        np.testing.assert_allclose(result[0.0]["theta"][0], 2.0, atol=0.05)


class TestApplyControlAdjustment:
    @pytest.mark.parametrize(
        "d_sq_val,diff_x_val,theta_val,expected_diff_y",
        [
            (0.0, 2.0, 1.5, 2.0),
            (1.0, 2.0, 1.5, 5.0),
        ],
    )
    def test_adjustment_applied_only_to_matching_d_level(self, d_sq_val, diff_x_val, theta_val, expected_diff_y):
        n = 10
        part = {
            "diff_y_1": np.ones(n) * 5.0,
            "d_sq": np.full(n, d_sq_val),
            "weight_gt": np.ones(n),
            "n_rows": n,
            "diff_x1_1": np.full(n, diff_x_val),
        }
        coefficients = {0.0: {"theta": np.array([theta_val])}}
        part = partition_apply_control_adjustment(part, abs_h=1, covariate_names=["x1"], coefficients=coefficients)
        np.testing.assert_allclose(part["diff_y_1"], expected_diff_y)


class TestReduceControlInfluenceSums:
    def test_merges_m_sum_in_sum_M_total(self):
        a = {
            "m_sum": {(0, 0.0): {1.0: 0.5, 2.0: 0.3}},
            "in_sum": {(0, 0.0, 2.0, 0.0): 1.0},
            "M_total": {(0, 0.0): 0.8},
        }
        b = {
            "m_sum": {(0, 0.0): {2.0: 0.7, 3.0: 0.1}},
            "in_sum": {(0, 0.0, 2.0, 0.0): 0.5, (0, 0.0, 3.0, 0.0): 0.2},
            "M_total": {(0, 0.0): 0.4},
        }
        merged = reduce_control_influence_sums(a, b)
        assert merged["m_sum"][(0, 0.0)][1.0] == 0.5
        assert merged["m_sum"][(0, 0.0)][2.0] == pytest.approx(1.0)
        assert merged["in_sum"][(0, 0.0, 2.0, 0.0)] == pytest.approx(1.5)
        assert merged["M_total"][(0, 0.0)] == pytest.approx(1.2)


def test_variance_part2_returns_per_unit_dict(large_partition):
    part = _run_horizon(large_partition, t_max=8.0)
    coefficients = {0.0: {"theta": np.array([1.0]), "inv_denom": np.array([[0.5]]), "useful": True}}
    result = partition_compute_variance_part2(
        part,
        _abs_h=1,
        covariate_names=["x1"],
        coefficients=coefficients,
        global_M_total={(0, 0.0): 5.0},
        n_groups=20,
    )
    assert isinstance(result, dict)


class TestCheckGroupLevelCovariates:
    @pytest.mark.parametrize(
        "values,should_vary",
        [
            ([0.5, 0.5, 0.8, 0.8], False),
            ([0.5, 0.6, 0.8, 0.8], True),
        ],
    )
    def test_detects_within_unit_variation(self, values, should_vary):
        part = {"gname": np.array([1.0, 1.0, 2.0, 2.0]), "x1": np.array(values)}
        varies = partition_check_group_level_covariates(part, ["x1"])
        assert ("x1" in varies) == should_vary

    def test_missing_column_flagged_as_varying(self):
        part = {"gname": np.array([1.0, 1.0])}
        varies = partition_check_group_level_covariates(part, ["missing_col"])
        assert "missing_col" in varies


class TestExtractHetData:
    def test_one_row_per_switcher_excluding_never_treated(self, basic_partition):
        basic_partition["t_max_by_group"] = basic_partition["T_g"].copy()
        rows = partition_extract_het_data(
            basic_partition, effects=2, het_covariates=[], trends_nonparam=None, trends_lin=False
        )
        gnames = {r["_gname"] for r in rows}
        assert gnames == {1.0, 2.0}

    def test_baseline_y_at_f_g_minus_1(self, basic_partition):
        basic_partition["t_max_by_group"] = basic_partition["T_g"].copy()
        rows = partition_extract_het_data(
            basic_partition, effects=1, het_covariates=[], trends_nonparam=None, trends_lin=False
        )
        for row in rows:
            mask = (basic_partition["gname"] == row["_gname"]) & (basic_partition["tname"] == row["_F_g"] - 1)
            np.testing.assert_equal(row["_Y_baseline"], basic_partition["y"][mask][0])

    def test_het_covariates_included(self, partition_with_covariates):
        partition_with_covariates["t_max_by_group"] = partition_with_covariates["T_g"].copy()
        rows = partition_extract_het_data(
            partition_with_covariates, effects=1, het_covariates=["x1"], trends_nonparam=None, trends_lin=False
        )
        for row in rows:
            assert not np.isnan(row["x1"])

    def test_trends_lin_adds_baseline_m2(self, basic_partition):
        basic_partition["t_max_by_group"] = basic_partition["T_g"].copy()
        rows = partition_extract_het_data(
            basic_partition, effects=1, het_covariates=[], trends_nonparam=None, trends_lin=True
        )
        for row in rows:
            assert "_Y_baseline_m2" in row

    @pytest.mark.parametrize("effects", [1, 2, 3])
    def test_effect_horizon_columns_created(self, basic_partition, effects):
        basic_partition["t_max_by_group"] = basic_partition["T_g"].copy()
        rows = partition_extract_het_data(
            basic_partition, effects=effects, het_covariates=[], trends_nonparam=None, trends_lin=False
        )
        for row in rows:
            for h in range(1, effects + 1):
                assert f"_Y_h{h}" in row


class TestPrepareHetSample:
    @pytest.fixture
    def het_df(self):
        return pl.DataFrame(
            {
                "_gname": [1.0, 2.0, 3.0],
                "_F_g": [3.0, 4.0, 3.0],
                "_S_g": [1.0, 1.0, -1.0],
                "_d_sq": [0.0, 0.0, 1.0],
                "_Y_baseline": [2.0, 3.0, 5.0],
                "_weight_gt": [1.0, 1.0, 1.0],
                "_t_max_by_group": [6.0, 6.0, 6.0],
                "_Y_h1": [4.0, 5.0, 3.0],
                "_Y_h2": [5.0, 6.0, 2.0],
            }
        )

    def test_returns_none_for_missing_horizon(self, het_df):
        assert prepare_het_sample(het_df, horizon=99, trends_lin=False) is None

    def test_filters_rows_past_t_max(self):
        het_df = pl.DataFrame(
            {
                "_gname": [1.0, 2.0],
                "_F_g": [3.0, 3.0],
                "_S_g": [1.0, 1.0],
                "_d_sq": [0.0, 0.0],
                "_Y_baseline": [2.0, 3.0],
                "_weight_gt": [1.0, 1.0],
                "_t_max_by_group": [3.0, 6.0],
                "_Y_h1": [4.0, 5.0],
                "_Y_h2": [5.0, 6.0],
            }
        )
        result = prepare_het_sample(het_df, horizon=2, trends_lin=False)
        assert len(result) == 1

    def test_prod_het_computed_correctly(self, het_df):
        result = prepare_het_sample(het_df, horizon=1, trends_lin=False)
        expected = np.array([1.0 * (4.0 - 2.0), 1.0 * (5.0 - 3.0), -1.0 * (3.0 - 5.0)])
        np.testing.assert_allclose(result["_prod_het"].to_numpy(), expected)

    @pytest.mark.parametrize("renamed_col", ["F_g", "S_g", "d_sq", "weight_gt"])
    def test_renames_columns(self, het_df, renamed_col):
        result = prepare_het_sample(het_df, horizon=1, trends_lin=False)
        assert renamed_col in result.columns

    def test_trends_lin_adjustment(self, het_df):
        het_df = het_df.with_columns(pl.Series("_Y_baseline_m2", [1.0, 2.0, 4.5]))
        result = prepare_het_sample(het_df, horizon=1, trends_lin=True)
        expected = np.array([1.0, 1.0, 2.5])
        np.testing.assert_allclose(result["_prod_het"].to_numpy(), expected)


class TestTrendsLinAccumulation:
    def test_cumulative_estimates(self):
        est, _, _ = apply_trends_lin_accumulation(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.1, 0.1]),
            [np.array([0.5, -0.5]), np.array([0.3, -0.3]), np.array([0.2, -0.2])],
            n_groups=2,
        )
        np.testing.assert_allclose(est, [1.0, 3.0, 6.0])

    def test_cumulative_influence_functions(self):
        _, _, ifs = apply_trends_lin_accumulation(
            np.array([1.0, 2.0]),
            np.array([0.1, 0.1]),
            [np.array([1.0, -1.0]), np.array([0.5, -0.5])],
            n_groups=2,
        )
        np.testing.assert_allclose(ifs[1], [1.5, -1.5])

    def test_se_recomputed_from_cumulative_influence(self):
        _, se, _ = apply_trends_lin_accumulation(
            np.array([1.0, 2.0]),
            np.array([0.1, 0.1]),
            [np.array([1.0, -1.0]), np.array([0.5, -0.5])],
            n_groups=2,
        )
        cumulative = np.array([1.5, -1.5])
        np.testing.assert_allclose(se[1], np.sqrt(np.sum(cumulative**2)) / 2)

    @pytest.mark.parametrize(
        "inf_funcs",
        [
            [],
            [np.array([np.nan, np.nan]), np.array([np.nan, np.nan])],
        ],
    )
    def test_degenerate_influence_returns_unchanged(self, inf_funcs):
        estimates = np.array([1.0, 2.0]) if inf_funcs else np.array([1.0])
        std_errors = np.array([0.1, 0.2]) if inf_funcs else np.array([0.1])
        est, _, _ = apply_trends_lin_accumulation(estimates, std_errors, inf_funcs, n_groups=2)
        np.testing.assert_array_equal(est, estimates)


def test_group_sums_and_scalars_matches_individual_calls(basic_partition):
    part = _run_horizon(basic_partition)
    gs, sc = partition_group_sums_and_scalars(part, abs_h=1)
    gs_standalone = partition_group_sums(part, abs_h=1)
    sc_standalone = partition_global_scalars(part, abs_h=1)
    assert gs.keys() == gs_standalone.keys()
    assert sc["switcher_gnames"] == sc_standalone["switcher_gnames"]
    np.testing.assert_allclose(sc["n_switchers_weighted"], sc_standalone["n_switchers_weighted"])


def test_influence_and_meta_returns_all_components(basic_partition):
    part, _, sc = _run_through_globals(basic_partition)
    (gname_if, _), sw_map, obs_count, dof = partition_influence_and_meta(
        part,
        abs_h=1,
        n_groups=3,
        n_switchers_weighted=sc["n_switchers_weighted"],
    )
    assert isinstance(gname_if, dict)
    assert isinstance(sw_map, dict)
    assert isinstance(obs_count, int)
    assert set(dof.keys()) == {"switcher", "control", "union"}


class TestEndToEndPipeline:
    def test_att_positive_for_positive_treatment_effect(self, large_partition):
        part, _, sc = _run_through_globals(large_partition, t_max=8.0)
        _, partial_sum = partition_compute_influence(
            part, abs_h=1, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"]
        )
        assert partial_sum / 20 > 0

    def test_variance_influence_produces_finite_positive_se(self, large_partition):
        part, _, sc = _run_through_globals(large_partition, t_max=8.0)
        partition_compute_influence(part, abs_h=1, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"])
        dof = partition_dof_stats(part, abs_h=1)
        var_if = partition_variance_influence(
            part, abs_h=1, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"], global_dof=dof
        )
        se = np.sqrt(np.sum(np.array(list(var_if.values())) ** 2)) / 20
        assert np.isfinite(se) and se > 0

    @pytest.mark.parametrize("abs_h", [1, 2, 3])
    def test_multi_horizon_estimates_all_finite(self, large_partition, abs_h):
        part_copy = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in large_partition.items()}
        part = _run_horizon(part_copy, abs_h=abs_h, t_max=8.0)
        gs = partition_group_sums(part, abs_h=abs_h)
        sc = partition_global_scalars(part, abs_h=abs_h)
        part = partition_apply_globals(part, abs_h=abs_h, global_group_sums=gs)
        _, psum = partition_compute_influence(
            part, abs_h=abs_h, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"]
        )
        assert np.isfinite(psum / 20)


class TestBuildPartitionArraysHetCovNewColumn:
    def test_het_covariate_new_column_extracted(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                "het_x": [0.7, 0.8],
            }
        )
        col_config = {
            "gname": "id",
            "tname": "time",
            "yname": "y",
            "dname": "d",
            "het_covariates": ["het_x"],
        }
        result = build_didinter_partition_arrays(pdf, col_config)
        np.testing.assert_array_equal(result["het_x"], [0.7, 0.8])

    def test_pandas_cluster_col_raw_path(self):
        pd = importorskip("pandas")
        pdf_pd = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "y": [1.0, 2.0],
                "d": [0, 1],
                "F_g": [2.0, 2.0],
                "L_g": [1.0, 1.0],
                "S_g": [1, 1],
                "d_sq": [0.0, 0.0],
                "weight_gt": [1.0, 1.0],
                "first_obs_by_gp": [1.0, 0.0],
                "T_g": [2.0, 2.0],
                "cl": ["A", "B"],
            }
        )
        col_config = {"gname": "id", "tname": "time", "yname": "y", "dname": "d", "cluster": "cl"}
        result = build_didinter_partition_arrays(pdf_pd, col_config)
        np.testing.assert_array_equal(result["cluster"], ["A", "B"])


class TestTrendVarsBranches:
    @pytest.fixture
    def partition_with_trend(self, basic_partition):
        basic_partition["trend1"] = basic_partition["tname"].copy()
        return basic_partition

    def test_group_sums_with_trend_vars(self, partition_with_trend):
        part = _run_horizon(partition_with_trend)
        gs = partition_group_sums(part, abs_h=1, trend_vars=["trend1"])
        for key in gs:
            assert len(key) == 3

    def test_apply_globals_with_trend_vars(self, partition_with_trend):
        part = _run_horizon(partition_with_trend)
        gs = partition_group_sums(part, abs_h=1, trend_vars=["trend1"])
        part = partition_apply_globals(part, abs_h=1, global_group_sums=gs, trend_vars=["trend1"])
        assert "n_control_1" in part

    def test_dof_stats_control_keys_extended_by_trend(self, partition_with_trend):
        part, _, _ = _run_through_globals(partition_with_trend)
        dof = partition_dof_stats(part, abs_h=1, trend_vars=["trend1"])
        for key in dof["control"]:
            assert len(key) == 3

    def test_variance_influence_with_trend_vars(self, partition_with_trend):
        part = _run_horizon(partition_with_trend)
        gs = partition_group_sums(part, abs_h=1, trend_vars=["trend1"])
        sc = partition_global_scalars(part, abs_h=1)
        part = partition_apply_globals(part, abs_h=1, global_group_sums=gs, trend_vars=["trend1"])
        partition_compute_influence(part, abs_h=1, n_groups=3, n_switchers_weighted=sc["n_switchers_weighted"])
        dof = partition_dof_stats(part, abs_h=1, trend_vars=["trend1"])
        var_if = partition_variance_influence(
            part,
            abs_h=1,
            n_groups=3,
            n_switchers_weighted=sc["n_switchers_weighted"],
            global_dof=dof,
            trend_vars=["trend1"],
        )
        assert isinstance(var_if, dict)


class TestControlInfluenceSumsFunction:
    def _setup_covariate_pipeline(self, rng):
        n_periods = 6
        n = 4 * n_periods
        gname = np.repeat([1.0, 2.0, 3.0, 4.0], n_periods)
        tname = np.tile(np.arange(1.0, n_periods + 1), 4)
        F_g = np.where(gname <= 2, 4.0, np.inf)
        S_g = np.where(gname <= 2, 1.0, 0.0)
        d_sq = np.zeros(n)
        L_g = np.where(gname <= 2, 3.0, np.inf)
        T_g = np.full(n, float(n_periods))
        x1 = rng.standard_normal(n)
        y = rng.standard_normal(n) * 0.1 + x1
        weight_gt = np.ones(n)
        d = np.zeros(n)
        d[(gname <= 2) & (tname >= 4)] = 1.0
        first_obs = np.zeros(n)
        for g in [1.0, 2.0, 3.0, 4.0]:
            first_obs[np.where(gname == g)[0][0]] = 1.0
        part = {
            "gname": gname,
            "tname": tname,
            "y": y,
            "d": d,
            "F_g": F_g,
            "L_g": L_g,
            "S_g": S_g,
            "d_sq": d_sq,
            "weight_gt": weight_gt,
            "first_obs_by_gp": first_obs,
            "T_g": T_g,
            "n_rows": n,
            "x1": x1,
        }

        part = _run_horizon(part, t_max=6.0)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["x1"])
        gs = partition_group_sums(part, abs_h=1)
        sc = partition_global_scalars(part, abs_h=1)
        part = partition_apply_globals(part, abs_h=1, global_group_sums=gs)

        gram = partition_control_gram(part, abs_h=1, covariate_names=["x1"])
        coefficients = solve_control_coefficients(gram, n_controls=1, n_groups=4)
        return part, coefficients, sc

    def test_produces_m_sum_and_in_sum(self, rng):
        part, coefficients, sc = self._setup_covariate_pipeline(rng)
        result = partition_control_influence_sums(
            part,
            abs_h=1,
            covariate_names=["x1"],
            coefficients=coefficients,
            n_groups=4,
            n_sw_weighted=sc["n_switchers_weighted"],
        )
        assert "m_sum" in result
        assert "in_sum" in result
        assert "M_total" in result

    def test_M_total_nonzero_for_useful_coefficients(self, rng):
        part, coefficients, sc = self._setup_covariate_pipeline(rng)
        useful_levels = [d for d, c in coefficients.items() if c.get("useful")]
        if not useful_levels:
            pytest.skip("No useful coefficient levels for this seed")
        result = partition_control_influence_sums(
            part,
            abs_h=1,
            covariate_names=["x1"],
            coefficients=coefficients,
            n_groups=4,
            n_sw_weighted=sc["n_switchers_weighted"],
        )
        total = sum(result["M_total"].values())
        assert total != 0.0


class TestDeltaAndVarianceWrapper:
    def test_returns_tuple_of_delta_d_and_variance(self, large_partition):
        part, _, sc = _run_through_globals(large_partition, t_max=8.0)
        partition_compute_influence(part, abs_h=1, n_groups=20, n_switchers_weighted=sc["n_switchers_weighted"])
        dof = partition_dof_stats(part, abs_h=1)

        dist_col = part.get("dist_1")
        gname = part["gname"]
        weight = part["weight_gt"]
        valid = (~np.isnan(dist_col)) & (dist_col == 1.0)
        sw_weights = dict(zip(gname[valid].tolist(), weight[valid].tolist(), strict=False))

        (dd_result, var_result) = partition_delta_and_variance(
            part,
            abs_h=1,
            horizon_type="effect",
            switcher_gnames_with_weight=sw_weights,
            n_groups=20,
            n_switchers_weighted=sc["n_switchers_weighted"],
            global_dof=dof,
        )
        contrib, w = dd_result
        assert isinstance(contrib, float)
        assert isinstance(w, float)
        assert isinstance(var_result, dict)


class TestTrendsLinClusteredVariance:
    def test_clustered_variance_path(self):
        rng = np.random.default_rng(42)
        n = 20
        estimates = np.array([1.0, 2.0])
        std_errors = np.array([0.1, 0.1])
        inf_funcs = [rng.standard_normal(n), rng.standard_normal(n)]
        cluster_ids = np.repeat(np.arange(5), 4)

        est, se, _ = apply_trends_lin_accumulation(
            estimates,
            std_errors,
            inf_funcs,
            n_groups=n,
            cluster_col="cl",
            cluster_ids=cluster_ids,
        )
        np.testing.assert_allclose(est, [1.0, 3.0])
        assert se[1] > 0


class TestExtractHetDataTrendsNonparam:
    def test_trends_nonparam_included_in_rows(self, basic_partition):
        basic_partition["t_max_by_group"] = basic_partition["T_g"].copy()
        basic_partition["tnp_var"] = basic_partition["tname"].copy()
        rows = partition_extract_het_data(
            basic_partition,
            effects=1,
            het_covariates=[],
            trends_nonparam=["tnp_var"],
            trends_lin=False,
        )
        for row in rows:
            assert "tnp_var" in row


class TestComputeVariancePart2WithUsefulCoefficients:
    def test_nonzero_adjustment_for_non_never_switchers(self, rng):
        n_periods = 6
        n = 4 * n_periods
        gname = np.repeat([1.0, 2.0, 3.0, 4.0], n_periods)
        tname = np.tile(np.arange(1.0, n_periods + 1), 4)
        F_g = np.where(gname <= 2, 4.0, np.inf)
        S_g = np.where(gname <= 2, 1.0, 0.0)
        d_sq = np.zeros(n)
        L_g = np.where(gname <= 2, 3.0, np.inf)
        T_g = np.full(n, float(n_periods))
        x1 = rng.standard_normal(n)
        y = rng.standard_normal(n) * 0.1 + x1
        weight_gt = np.ones(n)
        d = np.zeros(n)
        d[(gname <= 2) & (tname >= 4)] = 1.0
        first_obs = np.zeros(n)
        for g in [1.0, 2.0, 3.0, 4.0]:
            first_obs[np.where(gname == g)[0][0]] = 1.0
        part = {
            "gname": gname,
            "tname": tname,
            "y": y,
            "d": d,
            "F_g": F_g,
            "L_g": L_g,
            "S_g": S_g,
            "d_sq": d_sq,
            "weight_gt": weight_gt,
            "first_obs_by_gp": first_obs,
            "T_g": T_g,
            "n_rows": n,
            "x1": x1,
        }

        part = _run_horizon(part, t_max=6.0)
        part = partition_horizon_covariate_ops(part, abs_h=1, covariate_names=["x1"])
        gs = partition_group_sums(part, abs_h=1)
        part = partition_apply_globals(part, abs_h=1, global_group_sums=gs)

        gram = partition_control_gram(part, abs_h=1, covariate_names=["x1"])
        coefficients = solve_control_coefficients(gram, n_controls=1, n_groups=4)
        useful_levels = [d for d, c in coefficients.items() if c.get("useful")]
        if not useful_levels:
            pytest.skip("No useful coefficients for this seed")

        global_M_total = {(0, d_level): 5.0 for d_level in useful_levels}
        result = partition_compute_variance_part2(
            part,
            _abs_h=1,
            covariate_names=["x1"],
            coefficients=coefficients,
            global_M_total=global_M_total,
            n_groups=4,
        )
        assert len(result) > 0
