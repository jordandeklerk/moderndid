"""Tests for distributed DIDInter preprocessing functions."""

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.distributed._didinter_preprocess import (
    _extract_vars_from_formula,
    cap_effects_placebo,
    partition_extract_metadata,
    partition_preprocess_global,
    partition_preprocess_local,
    reduce_metadata,
    validate_distributed,
)

COL_CONFIG = {"gname": "id", "tname": "time", "yname": "y", "dname": "d"}


def _make_panel(n_units=5, n_periods=4, switch_at=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if switch_at is None:
        switch_at = {0: 3, 1: 3}

    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    d = np.zeros(len(ids))
    for unit, t in switch_at.items():
        d[(ids == unit) & (times >= t)] = 1.0
    y = rng.standard_normal(len(ids)) + 2.0 * d

    return pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})


def _preprocess_local(pdf, col_config=None, config_flags=None):
    if col_config is None:
        col_config = COL_CONFIG
    if config_flags is None:
        config_flags = {}
    return partition_preprocess_local(pdf, col_config, config_flags)


class TestExtractVarsFromFormula:
    @pytest.mark.parametrize(
        "formula,expected",
        [
            ("~x1 + x2", ["x1", "x2"]),
            ("~ x1 + x2 + x3", ["x1", "x2", "x3"]),
            ("~1", []),
            ("", []),
            (None, []),
            ("~x1", ["x1"]),
        ],
    )
    def test_parses_formula(self, formula, expected):
        assert _extract_vars_from_formula(formula) == expected


class TestPartitionPreprocessLocal:
    def test_empty_dataframe_returned_unchanged(self):
        empty = pl.DataFrame({"id": [], "time": [], "y": [], "d": []}).cast(
            {"id": pl.Int64, "time": pl.Int64, "y": pl.Float64, "d": pl.Float64}
        )
        result = _preprocess_local(empty)
        assert len(result) == 0

    def test_creates_required_columns(self):
        pdf = _make_panel()
        result = _preprocess_local(pdf)
        for col in ("F_g", "d_sq", "S_g", "L_g", "weight_gt", "first_obs_by_gp", "t_max_by_group", "d_fg"):
            assert col in result.columns, f"Missing column: {col}"

    def test_f_g_inf_for_never_switchers(self):
        pdf = _make_panel(n_units=4, switch_at={0: 3})
        result = _preprocess_local(pdf)
        never_switched = result.filter(pl.col("id") > 0)
        assert (never_switched["F_g"] == float("inf")).all()

    def test_f_g_correct_for_switchers(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3, 1: 2})
        result = _preprocess_local(pdf)
        unit_0 = result.filter(pl.col("id") == 0)["F_g"].unique().item()
        unit_1 = result.filter(pl.col("id") == 1)["F_g"].unique().item()
        assert unit_0 == 3.0
        assert unit_1 == 2.0

    def test_s_g_positive_for_in_switcher(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        result = _preprocess_local(pdf)
        s_g_unit0 = result.filter(pl.col("id") == 0)["S_g"].unique().item()
        assert s_g_unit0 == 1

    def test_s_g_negative_for_out_switcher(self):
        ids = np.repeat([0, 1], 4)
        times = np.tile([1, 2, 3, 4], 2)
        d = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        y = np.ones(8)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf)
        s_g_unit0 = result.filter(pl.col("id") == 0)["S_g"].unique().item()
        assert s_g_unit0 == -1

    def test_s_g_zero_for_never_switcher(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        result = _preprocess_local(pdf)
        s_g_unit2 = result.filter(pl.col("id") == 2)["S_g"].unique().item()
        assert s_g_unit2 == 0

    def test_d_sq_is_baseline_treatment(self):
        ids = np.repeat([0, 1], 4)
        times = np.tile([1, 2, 3, 4], 2)
        d = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        y = np.ones(8)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf)
        d_sq_unit0 = result.filter(pl.col("id") == 0)["d_sq"].unique().item()
        d_sq_unit1 = result.filter(pl.col("id") == 1)["d_sq"].unique().item()
        assert d_sq_unit0 == 0.0
        assert d_sq_unit1 == 1.0

    def test_l_g_correct_for_switcher(self):
        pdf = _make_panel(n_units=3, n_periods=6, switch_at={0: 3})
        result = _preprocess_local(pdf)
        l_g = result.filter(pl.col("id") == 0)["L_g"].unique().item()
        assert l_g == 4.0

    def test_l_g_zero_for_never_switcher(self):
        pdf = _make_panel(n_units=3, switch_at={})
        result = _preprocess_local(pdf)
        l_g_vals = result["L_g"].unique().to_list()
        assert all(v == 0.0 for v in l_g_vals)

    def test_bidirectional_switchers_removed_by_default(self):
        ids = np.repeat([0, 1, 2], 5)
        times = np.tile([1, 2, 3, 4, 5], 3)
        d = np.zeros(15, dtype=float)
        d[ids == 0] = [1.0, 2.0, 0.0, 1.0, 0.0]
        d[(ids == 1) & (times >= 3)] = 1.0
        y = np.ones(15)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf)
        unit_0_rows = result.filter(pl.col("id") == 0)
        assert len(unit_0_rows) < 5
        unit_1_rows = result.filter(pl.col("id") == 1)
        assert len(unit_1_rows) == 5

    def test_bidirectional_switchers_kept_when_flag_set(self):
        ids = np.repeat([0, 1], 5)
        times = np.tile([1, 2, 3, 4, 5], 2)
        d = np.zeros(10)
        d[(ids == 0) & (times >= 2) & (times <= 3)] = 1.0
        y = np.ones(10)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf, config_flags={"keep_bidirectional_switchers": True})
        assert 0 in result["id"].unique().to_list()

    @pytest.mark.parametrize(
        "mode,expected_s_g_values",
        [
            ("in", {0, 1}),
            ("out", {0, -1}),
        ],
    )
    def test_switchers_mode_filters_direction(self, mode, expected_s_g_values):
        ids = np.repeat([0, 1, 2], 4)
        times = np.tile([1, 2, 3, 4], 3)
        d = np.zeros(12)
        d[(ids == 0) & (times >= 3)] = 1.0
        d[(ids == 1) & (times < 3)] = 1.0
        y = np.ones(12)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf, config_flags={"switchers": mode})
        s_g_values = set(result["S_g"].unique().to_list())
        assert s_g_values == expected_s_g_values

    def test_weight_gt_defaults_to_one(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        result = _preprocess_local(pdf)
        np.testing.assert_array_equal(result["weight_gt"].to_numpy(), 1.0)

    def test_custom_weights_applied(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        pdf = pdf.with_columns(pl.Series("w", np.full(len(pdf), 2.5)))
        result = _preprocess_local(pdf, config_flags={"weightsname": "w"})
        np.testing.assert_allclose(result["weight_gt"].to_numpy(), 2.5)

    def test_weight_gt_zeroed_for_missing_y(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [1, 2, 3],
                "y": [1.0, None, 3.0],
                "d": [0.0, 0.0, 0.0],
            }
        )
        result = _preprocess_local(pdf)
        wt = result.sort("time")["weight_gt"].to_list()
        assert wt[1] == 0.0
        assert wt[0] == 1.0

    def test_first_obs_by_gp_marks_first_period(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        result = _preprocess_local(pdf).sort(["id", "time"])
        for uid in result["id"].unique().to_list():
            unit_rows = result.filter(pl.col("id") == uid).sort("time")
            assert unit_rows["first_obs_by_gp"].to_list()[0] == 1
            assert all(v == 0 for v in unit_rows["first_obs_by_gp"].to_list()[1:])

    def test_t_max_by_group_correct(self):
        pdf = _make_panel(n_units=2, n_periods=5, switch_at={0: 3})
        result = _preprocess_local(pdf)
        for uid in result["id"].unique().to_list():
            t_max = result.filter(pl.col("id") == uid)["t_max_by_group"].unique().item()
            assert t_max == 5.0

    def test_extra_columns_kept_via_xformla(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        pdf = pdf.with_columns(pl.Series("x1", np.ones(len(pdf))))
        result = _preprocess_local(pdf, config_flags={"xformla": "~x1"})
        assert "x1" in result.columns

    def test_cluster_column_kept(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        pdf = pdf.with_columns(pl.Series("cl", ["A"] * len(pdf)))
        col_config = {**COL_CONFIG, "cluster": "cl"}
        result = _preprocess_local(pdf, col_config=col_config)
        assert "cl" in result.columns

    def test_trends_nonparam_column_kept(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        pdf = pdf.with_columns(pl.Series("region", [1] * len(pdf)))
        result = _preprocess_local(pdf, config_flags={"trends_nonparam": ["region"]})
        assert "region" in result.columns

    def test_het_covariates_column_kept(self):
        pdf = _make_panel(n_units=2, switch_at={0: 3})
        pdf = pdf.with_columns(pl.Series("het_x", np.ones(len(pdf))))
        col_config = {**COL_CONFIG, "het_covariates": ["het_x"]}
        result = _preprocess_local(pdf, col_config=col_config)
        assert "het_x" in result.columns

    def test_pandas_input_converted_to_polars(self):
        importorskip("pandas")
        pdf_pl = _make_panel(n_units=2, switch_at={0: 3})
        pdf_pd = pdf_pl.to_pandas()
        result = _preprocess_local(pdf_pd)
        assert isinstance(result, pl.DataFrame)
        assert "F_g" in result.columns

    def test_string_typed_columns_cast_to_float(self):
        pdf = pl.DataFrame(
            {
                "id": ["1", "1", "2", "2"],
                "time": [1, 2, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0],
                "d": [0.0, 1.0, 0.0, 0.0],
            }
        )
        result = _preprocess_local(pdf)
        assert result["id"].dtype == pl.Float64

    def test_drop_missing_preswitch(self):
        pdf = pl.DataFrame(
            {
                "id": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y": [1.0, 2.0, 3.0, 4.0],
                "d": [0.0, None, None, 1.0],
            }
        )
        n_before = len(_preprocess_local(pdf))
        result = _preprocess_local(pdf, config_flags={"drop_missing_preswitch": True})
        assert len(result) <= n_before


class TestPartitionExtractMetadata:
    def _preprocess_and_extract(self, pdf, config_flags=None):
        if config_flags is None:
            config_flags = {}
        local = _preprocess_local(pdf, config_flags=config_flags)
        meta_flags = {**config_flags, "gname": "id", "tname": "time"}
        return partition_extract_metadata(local, meta_flags)

    def test_empty_dataframe_returns_defaults(self):
        empty = pl.DataFrame({"id": [], "time": [], "y": [], "d": []}).cast(
            {"id": pl.Int64, "time": pl.Int64, "y": pl.Float64, "d": pl.Float64}
        )
        meta = partition_extract_metadata(empty, {"gname": "id", "tname": "time"})
        assert meta["weight_sum"] == 0.0
        assert meta["n_switchers"] == 0
        assert meta["t_min"] == float("inf")

    def test_weight_sum_equals_row_count_for_unit_weights(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert meta["weight_sum"] == pytest.approx(meta["weight_count"])

    def test_unique_times_correct(self):
        pdf = _make_panel(n_units=3, n_periods=5, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert meta["unique_times"] == {1.0, 2.0, 3.0, 4.0, 5.0}

    def test_t_min_and_t_max(self):
        pdf = _make_panel(n_units=3, n_periods=6, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert meta["t_min"] == 1.0
        assert meta["t_max"] == 6.0

    def test_n_switchers_and_never_switchers_counted(self):
        pdf = _make_panel(n_units=5, switch_at={0: 3, 1: 4})
        meta = self._preprocess_and_extract(pdf)
        assert meta["n_switchers"] == 2
        assert meta["n_never_switchers"] == 3

    def test_max_effects_available(self):
        pdf = _make_panel(n_units=3, n_periods=6, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert meta["max_effects_available"] == 4

    def test_max_placebo_available(self):
        pdf = _make_panel(n_units=3, n_periods=6, switch_at={0: 4})
        meta = self._preprocess_and_extract(pdf)
        assert meta["max_placebo_available"] == 2

    def test_fg_variation_has_multiple_values_when_staggered(self):
        pdf = _make_panel(n_units=4, n_periods=5, switch_at={0: 3, 1: 4})
        meta = self._preprocess_and_extract(pdf)
        all_vals = set()
        for v in meta["fg_variation"].values():
            all_vals |= v
        assert len(all_vals) > 1

    def test_controls_time_nonempty_with_never_switchers(self):
        pdf = _make_panel(n_units=4, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert len(meta["controls_time"]) > 0

    def test_all_gnames_first_obs_matches_unique_units(self):
        pdf = _make_panel(n_units=4, switch_at={0: 3})
        meta = self._preprocess_and_extract(pdf)
        assert len(meta["all_gnames_first_obs"]) == 4

    def test_no_switchers_gives_zero_effects_and_placebo(self):
        pdf = _make_panel(n_units=3, switch_at={})
        meta = self._preprocess_and_extract(pdf)
        assert meta["max_effects_available"] == 0
        assert meta["max_placebo_available"] == 0

    def test_pandas_input_handled(self):
        importorskip("pandas")
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        local = _preprocess_local(pdf)
        meta = partition_extract_metadata(local.to_pandas(), {"gname": "id", "tname": "time"})
        assert meta["n_switchers"] == 1


class TestReduceMetadata:
    def test_sums_weights_and_counts(self):
        a = _make_empty_meta(weight_sum=10.0, weight_count=5, t_min=1.0, t_max=4.0)
        b = _make_empty_meta(weight_sum=6.0, weight_count=3, t_min=2.0, t_max=5.0)
        merged = reduce_metadata(a, b)
        assert merged["weight_sum"] == 16.0
        assert merged["weight_count"] == 8

    def test_takes_global_min_max_times(self):
        a = _make_empty_meta(t_min=2.0, t_max=5.0)
        b = _make_empty_meta(t_min=1.0, t_max=7.0)
        merged = reduce_metadata(a, b)
        assert merged["t_min"] == 1.0
        assert merged["t_max"] == 7.0

    def test_unions_unique_times_and_d_sq(self):
        a = _make_empty_meta(unique_times={1.0, 2.0}, unique_d_sq={0.0})
        b = _make_empty_meta(unique_times={2.0, 3.0}, unique_d_sq={0.0, 1.0})
        merged = reduce_metadata(a, b)
        assert merged["unique_times"] == {1.0, 2.0, 3.0}
        assert merged["unique_d_sq"] == {0.0, 1.0}

    def test_unions_fg_variation(self):
        a = _make_empty_meta(fg_variation={(0.0,): {3.0, 0.0}})
        b = _make_empty_meta(fg_variation={(0.0,): {4.0, 0.0}})
        merged = reduce_metadata(a, b)
        assert merged["fg_variation"][(0.0,)] == {3.0, 4.0, 0.0}

    def test_takes_max_fg_trunc_max(self):
        a = _make_empty_meta(fg_trunc_max={(0.0,): 5.0})
        b = _make_empty_meta(fg_trunc_max={(0.0,): 7.0})
        merged = reduce_metadata(a, b)
        assert merged["fg_trunc_max"][(0.0,)] == 7.0

    def test_sums_switcher_counts(self):
        a = _make_empty_meta(n_switchers=2, n_never_switchers=3)
        b = _make_empty_meta(n_switchers=1, n_never_switchers=4)
        merged = reduce_metadata(a, b)
        assert merged["n_switchers"] == 3
        assert merged["n_never_switchers"] == 7

    def test_takes_max_effects_and_placebo(self):
        a = _make_empty_meta(max_effects_available=3, max_placebo_available=1)
        b = _make_empty_meta(max_effects_available=5, max_placebo_available=2)
        merged = reduce_metadata(a, b)
        assert merged["max_effects_available"] == 5
        assert merged["max_placebo_available"] == 2


def _make_empty_meta(**overrides):
    base = {
        "weight_sum": 0.0,
        "weight_count": 0,
        "fg_variation": {},
        "controls_time": set(),
        "unique_times": set(),
        "unique_d_sq": set(),
        "t_min": float("inf"),
        "t_max": float("-inf"),
        "n_switchers": 0,
        "n_never_switchers": 0,
        "all_gnames_first_obs": set(),
        "max_effects_available": 0,
        "max_placebo_available": 0,
        "fg_trunc_max": {},
    }
    base.update(overrides)
    return base


class TestPartitionPreprocessGlobal:
    def _run_full_pipeline(self, pdf, col_config=None, config_flags=None):
        if col_config is None:
            col_config = COL_CONFIG
        if config_flags is None:
            config_flags = {}
        local = partition_preprocess_local(pdf, col_config, config_flags)
        meta_flags = {**config_flags, "gname": "id", "tname": "time"}
        meta = partition_extract_metadata(local, meta_flags)
        return partition_preprocess_global(local, meta, col_config, config_flags)

    def test_empty_dataframe_returned_unchanged(self):
        empty = pl.DataFrame({"id": [], "time": [], "y": [], "d": []}).cast(
            {"id": pl.Int64, "time": pl.Int64, "y": pl.Float64, "d": pl.Float64}
        )
        result = partition_preprocess_global(empty, _make_empty_meta(), COL_CONFIG, {})
        assert len(result) == 0

    def test_creates_T_g_column(self):
        pdf = _make_panel(n_units=4, n_periods=5, switch_at={0: 3, 1: 4})
        result = self._run_full_pipeline(pdf)
        assert "T_g" in result.columns
        assert result["T_g"].null_count() == 0

    def test_weight_gt_normalized(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        result = self._run_full_pipeline(pdf)
        mean_wt = result["weight_gt"].mean()
        np.testing.assert_allclose(mean_wt, 1.0, atol=0.2)

    def test_panel_balanced_after_global(self):
        pdf = _make_panel(n_units=3, n_periods=4, switch_at={0: 3})
        result = self._run_full_pipeline(pdf)
        counts = result.group_by("id").agg(pl.len().alias("n")).sort("id")
        assert counts["n"].n_unique() == 1

    def test_first_obs_by_gp_recomputed(self):
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        result = self._run_full_pipeline(pdf).sort(["id", "time"])
        for uid in result["id"].unique().to_list():
            unit_rows = result.filter(pl.col("id") == uid).sort("time")
            assert unit_rows["first_obs_by_gp"].to_list()[0] == 1
            assert all(v == 0 for v in unit_rows["first_obs_by_gp"].to_list()[1:])

    def test_same_switchers_creates_valid_column(self):
        pdf = _make_panel(n_units=4, n_periods=6, switch_at={0: 3, 1: 4})
        result = self._run_full_pipeline(pdf, config_flags={"same_switchers": True, "effects": 2})
        assert "same_switcher_valid" in result.columns

    def test_same_switchers_pl_creates_valid_column(self):
        pdf = _make_panel(n_units=4, n_periods=6, switch_at={0: 4, 1: 5})
        result = self._run_full_pipeline(pdf, config_flags={"same_switchers_pl": True, "placebo": 2})
        assert "same_switcher_valid" in result.columns

    def test_trends_lin_first_differences_outcome(self):
        rng = np.random.default_rng(123)
        pdf = _make_panel(n_units=4, n_periods=6, switch_at={0: 4, 1: 5}, rng=rng)
        result = self._run_full_pipeline(pdf, config_flags={"trends_lin": True})
        assert len(result) > 0
        assert result["time"].min() > 1

    def test_trends_lin_with_xformla_differences_covariates(self):
        rng = np.random.default_rng(123)
        pdf = _make_panel(n_units=4, n_periods=6, switch_at={0: 4, 1: 5}, rng=rng)
        pdf = pdf.with_columns(pl.Series("x1", rng.standard_normal(len(pdf))))
        result = self._run_full_pipeline(pdf, config_flags={"trends_lin": True, "xformla": "~x1"})
        assert "x1" in result.columns

    def test_continuous_treatment_creates_polynomial_columns(self):
        ids = np.repeat([0, 1, 2, 3], 4)
        times = np.tile([1, 2, 3, 4], 4)
        d = np.zeros(16)
        d[(ids == 0) & (times >= 3)] = 2.0
        d[(ids == 1) & (times >= 3)] = 3.0
        y = np.ones(16) + d
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = self._run_full_pipeline(pdf, config_flags={"continuous": 2})
        assert "d_sq_1" in result.columns
        assert "d_sq_2" in result.columns

    def test_d_fg_filled_across_balanced_panel(self):
        pdf = _make_panel(n_units=4, n_periods=5, switch_at={0: 3, 1: 4})
        result = self._run_full_pipeline(pdf)
        if "d_fg" in result.columns:
            for uid in result.filter(pl.col("F_g") != float("inf"))["id"].unique().to_list():
                unit_dfg = result.filter(pl.col("id") == uid)["d_fg"]
                assert unit_dfg.null_count() == 0

    def test_t_max_by_group_filled_across_balanced_panel(self):
        pdf = _make_panel(n_units=3, n_periods=5, switch_at={0: 3})
        result = self._run_full_pipeline(pdf)
        if "t_max_by_group" in result.columns:
            assert result["t_max_by_group"].null_count() == 0

    def test_pandas_input_handled(self):
        importorskip("pandas")
        pdf = _make_panel(n_units=3, switch_at={0: 3})
        local = _preprocess_local(pdf)
        meta_flags = {"gname": "id", "tname": "time"}
        meta = partition_extract_metadata(local, meta_flags)
        result = partition_preprocess_global(local.to_pandas(), meta, COL_CONFIG, {})
        assert isinstance(result, pl.DataFrame)


class TestCapEffectsPlacebo:
    @pytest.mark.parametrize(
        "req_eff,max_eff,expected_eff",
        [
            (3, 5, 3),
            (10, 5, 5),
        ],
    )
    def test_effects_capped(self, req_eff, max_eff, expected_eff):
        config = {"effects": req_eff, "placebo": 0}
        meta = {"max_effects_available": max_eff, "max_placebo_available": 10}
        if req_eff > max_eff:
            with pytest.warns(UserWarning, match="effects="):
                eff, _ = cap_effects_placebo(config, meta)
        else:
            eff, _ = cap_effects_placebo(config, meta)
        assert eff == expected_eff
        assert config["effects"] == expected_eff

    @pytest.mark.parametrize(
        "req_plac,max_plac,expected_plac",
        [
            (2, 5, 2),
            (8, 3, 3),
        ],
    )
    def test_placebo_capped(self, req_plac, max_plac, expected_plac):
        config = {"effects": 1, "placebo": req_plac}
        meta = {"max_effects_available": 10, "max_placebo_available": max_plac}
        if req_plac > max_plac:
            with pytest.warns(UserWarning, match="placebo="):
                _, plac = cap_effects_placebo(config, meta)
        else:
            _, plac = cap_effects_placebo(config, meta)
        assert plac == expected_plac
        assert config["placebo"] == expected_plac


class TestTrendsNonparamPaths:
    def test_extract_metadata_with_trends_nonparam(self):
        rng = np.random.default_rng(99)
        pdf = _make_panel(n_units=4, n_periods=5, switch_at={0: 3, 1: 4}, rng=rng)
        pdf = pdf.with_columns(pl.Series("region", np.tile(["east", "west"], len(pdf) // 2)))
        local = _preprocess_local(pdf, config_flags={"trends_nonparam": ["region"]})
        meta = partition_extract_metadata(local, {"gname": "id", "tname": "time", "trends_nonparam": ["region"]})
        assert len(meta["fg_variation"]) > 0
        assert any(len(k) == 2 for k in meta["fg_variation"])

    def test_global_preprocess_with_trends_nonparam(self):
        rng = np.random.default_rng(99)
        pdf = _make_panel(n_units=4, n_periods=5, switch_at={0: 3, 1: 4}, rng=rng)
        pdf = pdf.with_columns(pl.Series("region", np.tile(["east", "west"], len(pdf) // 2)))
        local = _preprocess_local(pdf, config_flags={"trends_nonparam": ["region"]})
        meta_flags = {"gname": "id", "tname": "time", "trends_nonparam": ["region"]}
        meta = partition_extract_metadata(local, meta_flags)
        result = partition_preprocess_global(
            local,
            meta,
            COL_CONFIG,
            {"trends_nonparam": ["region"]},
        )
        assert len(result) > 0

    def test_all_bidirectional_removed_gives_empty(self):
        ids = np.repeat([0], 4)
        times = np.array([1, 2, 3, 4])
        d = np.array([1.0, 2.0, 0.0, 1.0])
        y = np.ones(4)
        pdf = pl.DataFrame({"id": ids, "time": times, "y": y, "d": d})
        result = _preprocess_local(pdf)
        assert len(result) < 4


class TestValidateDistributed:
    def test_passes_with_all_columns_present(self):
        validate_distributed(["y", "time", "id", "d"], COL_CONFIG, {})

    @pytest.mark.parametrize("missing_col", ["y", "time", "id", "d"])
    def test_raises_for_missing_required_column(self, missing_col):
        cols = {"y", "time", "id", "d"} - {missing_col}
        with pytest.raises(ValueError, match="must be a column"):
            validate_distributed(list(cols), COL_CONFIG, {})

    def test_raises_for_missing_weights_column(self):
        with pytest.raises(ValueError, match="weightsname"):
            validate_distributed(["y", "time", "id", "d"], COL_CONFIG, {"weightsname": "w"})

    def test_raises_for_missing_cluster_column(self):
        col_config = {**COL_CONFIG, "cluster": "cl"}
        with pytest.raises(ValueError, match="cluster"):
            validate_distributed(["y", "time", "id", "d"], col_config, {})

    def test_raises_for_missing_xformla_variable(self):
        with pytest.raises(ValueError, match="x1"):
            validate_distributed(["y", "time", "id", "d"], COL_CONFIG, {"xformla": "~x1 + x2"})

    def test_passes_with_xformla_tilde_one(self):
        validate_distributed(["y", "time", "id", "d"], COL_CONFIG, {"xformla": "~1"})

    def test_passes_with_all_optional_columns_present(self):
        col_config = {**COL_CONFIG, "cluster": "cl"}
        validate_distributed(
            ["y", "time", "id", "d", "w", "cl", "x1"],
            col_config,
            {"weightsname": "w", "xformla": "~x1"},
        )
