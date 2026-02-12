"""Tests for partition utilities."""

import polars as pl
from moderndid.core.partition import build_gt_partitions, concat_partitions


class TestBuildGtPartitions:
    def test_returns_dict(self, sample_data):
        result = build_gt_partitions(sample_data, "group", "time")
        assert isinstance(result, dict)

    def test_correct_keys(self, sample_data):
        result = build_gt_partitions(sample_data, "group", "time")
        expected_keys = {(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)}
        assert set(result.keys()) == expected_keys

    def test_correct_row_counts(self, sample_data):
        result = build_gt_partitions(sample_data, "group", "time")
        assert len(result[(1, 1)]) == 2
        assert len(result[(0, 2)]) == 2

    def test_partition_values(self, sample_data):
        result = build_gt_partitions(sample_data, "group", "time")
        part = result[(1, 1)]
        assert (part["group"] == 1).all()
        assert (part["time"] == 1).all()


class TestConcatPartitions:
    def test_single_key(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        result = concat_partitions(partitions, [1], [1])
        assert len(result) == 2
        assert (result["group"] == 1).all()
        assert (result["time"] == 1).all()

    def test_multiple_groups(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        result = concat_partitions(partitions, [1, 0], [1, 2])
        assert len(result) == 8

    def test_missing_keys_returns_none(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        result = concat_partitions(partitions, [99], [99])
        assert result is None

    def test_partial_missing_keys(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        result = concat_partitions(partitions, [1, 99], [1])
        assert len(result) == 2

    def test_round_trip(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        all_groups = sorted(set(g for g, _ in partitions))
        all_times = sorted(set(t for _, t in partitions))
        result = concat_partitions(partitions, all_groups, all_times)
        assert len(result) == len(sample_data)
        assert set(result.columns) == set(sample_data.columns)

    def test_empty_group_vals(self, sample_data):
        partitions = build_gt_partitions(sample_data, "group", "time")
        result = concat_partitions(partitions, [], [1, 2])
        assert result is None


class TestWithInfGroups:
    def test_inf_group_values(self):
        data = pl.DataFrame(
            {
                "group": [1.0, 1.0, float("inf"), float("inf")],
                "time": [1, 2, 1, 2],
                "y": [10, 11, 20, 21],
            }
        )
        partitions = build_gt_partitions(data, "group", "time")
        assert (1.0, 1) in partitions
        assert (float("inf"), 1) in partitions
        result = concat_partitions(partitions, [1.0, float("inf")], [1])
        assert len(result) == 2
