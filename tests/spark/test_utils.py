"""Tests for distributed utility functions."""

import numpy as np
import pytest

from moderndid.spark._utils import auto_tune_partitions, is_spark_dataframe, validate_spark_input


def test_auto_tune_no_adjustment():
    result = auto_tune_partitions(n_default=4, n_units=100, k=2)
    assert result == 4


def test_auto_tune_increases_for_large():
    result = auto_tune_partitions(n_default=4, n_units=int(1e9), k=10)
    assert result > 4


@pytest.mark.parametrize("n_default", [1, 4, 16])
def test_auto_tune_never_below_default(n_default):
    result = auto_tune_partitions(n_default=n_default, n_units=100, k=2)
    assert result >= n_default


def test_auto_tune_returns_int():
    result = auto_tune_partitions(n_default=4, n_units=int(1e8), k=5)
    assert isinstance(result, int)


def test_auto_tune_custom_target_bytes():
    result = auto_tune_partitions(n_default=1, n_units=1000, k=2, target_bytes=100)
    assert result > 1


@pytest.mark.parametrize("obj", [np.array([1, 2, 3]), [1, 2, 3], None])
def test_is_spark_dataframe_non_spark(obj):
    assert is_spark_dataframe(obj) is False


def test_is_spark_dataframe_spark_df(spark_session):
    sdf = spark_session.createDataFrame([(1, 2)], ["a", "b"])
    assert is_spark_dataframe(sdf) is True


def test_validate_spark_input_passes(spark_session):
    sdf = spark_session.createDataFrame([(1, 2)], ["a", "b"])
    validate_spark_input(sdf, ["a", "b"])


def test_validate_spark_input_missing_raises(spark_session):
    sdf = spark_session.createDataFrame([(1,)], ["a"])
    with pytest.raises(ValueError, match="Columns not found"):
        validate_spark_input(sdf, ["a", "z"])
