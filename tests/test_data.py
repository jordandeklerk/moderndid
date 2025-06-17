"""Test the datasets module."""

import numpy as np
import pandas as pd

from pydid.data import load_nsw


def test_load_nsw():
    nsw_data = load_nsw()

    assert isinstance(nsw_data, pd.DataFrame)
    assert nsw_data.shape == (32834, 16)

    expected_columns = [
        "id",
        "year",
        "experimental",
        "re",
        "age",
        "educ",
        "black",
        "married",
        "nodegree",
        "hisp",
        "re74",
    ]
    for col in expected_columns:
        assert col in nsw_data.columns

    assert set(nsw_data["year"].unique()) == {1975, 1978}
    assert set(nsw_data["experimental"].unique()) == {0, 1}

    assert nsw_data["experimental"].sum() > 0
    assert (nsw_data["experimental"] == 0).sum() > 0


def test_load_nsw_data_integrity():
    nsw_data = load_nsw()

    key_columns = ["id", "year", "experimental"]
    for col in key_columns:
        assert nsw_data[col].notna().all()

    assert nsw_data["id"].dtype in [np.int64, np.int32]
    assert nsw_data["year"].dtype in [np.int64, np.int32]
    assert nsw_data["experimental"].dtype in [np.int64, np.int32]
    assert nsw_data["re"].dtype in [np.float64, np.float32]

    id_counts = nsw_data["id"].value_counts()
    assert (id_counts == 2).all()


def test_import_from_package():
    import pydid

    assert hasattr(pydid, "data")
    assert hasattr(pydid.data, "load_nsw")

    nsw_data = pydid.data.load_nsw()

    assert isinstance(nsw_data, pd.DataFrame)
    assert nsw_data.shape[0] > 0


def test_load_nsw_returns_copy():
    nsw_data1 = load_nsw()
    nsw_data2 = load_nsw()

    nsw_data1["test_column"] = 1

    assert "test_column" not in nsw_data2.columns
