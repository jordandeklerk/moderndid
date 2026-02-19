"""Tests for distributed utility functions."""

import numpy as np
import pytest

from moderndid.dask._utils import auto_tune_partitions, is_dask_collection, validate_dask_input


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
def test_is_dask_collection_non_dask(obj):
    assert is_dask_collection(obj) is False


def test_is_dask_collection_dask_df():
    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    ddf = dd.from_pandas(pd.DataFrame({"a": [1, 2]}), npartitions=1)
    assert is_dask_collection(ddf) is True


def test_validate_dask_input_passes():
    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    ddf = dd.from_pandas(pd.DataFrame({"a": [1], "b": [2]}), npartitions=1)
    validate_dask_input(ddf, ["a", "b"])


def test_validate_dask_input_missing_raises():
    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    ddf = dd.from_pandas(pd.DataFrame({"a": [1]}), npartitions=1)
    with pytest.raises(ValueError, match="Columns not found"):
        validate_dask_input(ddf, ["a", "z"])
