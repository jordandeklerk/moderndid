"""Test the datasets."""

import numpy as np
import pandas as pd

from pydid.data import load_mpdta, load_nsw


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


def test_load_mpdta():
    mpdta_data = load_mpdta()

    assert isinstance(mpdta_data, pd.DataFrame)
    assert mpdta_data.shape == (2500, 6)

    expected_columns = [
        "year",
        "countyreal",
        "lpop",
        "lemp",
        "first.treat",
        "treat",
    ]
    assert list(mpdta_data.columns) == expected_columns

    assert set(mpdta_data["year"].unique()) == {2003, 2004, 2005, 2006, 2007}
    assert set(mpdta_data["treat"].unique()) == {0, 1}
    assert set(mpdta_data["first.treat"].unique()) == {0, 2004, 2006, 2007}


def test_load_mpdta_data_integrity():
    mpdta_data = load_mpdta()

    key_columns = ["year", "countyreal", "lpop", "lemp", "first.treat", "treat"]
    for col in key_columns:
        assert mpdta_data[col].notna().all()

    assert mpdta_data["year"].dtype in [np.int64, np.int32]
    assert mpdta_data["countyreal"].dtype in [np.int64, np.int32]
    assert mpdta_data["lpop"].dtype in [np.float64, np.float32]
    assert mpdta_data["lemp"].dtype in [np.float64, np.float32]
    assert mpdta_data["first.treat"].dtype in [np.int64, np.int32]
    assert mpdta_data["treat"].dtype in [np.int64, np.int32]

    county_counts = mpdta_data["countyreal"].value_counts()
    assert (county_counts == 5).all()

    for county in mpdta_data["countyreal"].unique():
        county_data = mpdta_data[mpdta_data["countyreal"] == county]
        assert len(county_data["first.treat"].unique()) == 1


def test_import_mpdta_from_package():
    import pydid

    assert hasattr(pydid, "load_mpdta")
    assert hasattr(pydid.data, "load_mpdta")

    mpdta_data = pydid.load_mpdta()

    assert isinstance(mpdta_data, pd.DataFrame)
    assert mpdta_data.shape[0] == 2500


def test_load_mpdta_returns_copy():
    mpdta_data1 = load_mpdta()
    mpdta_data2 = load_mpdta()

    mpdta_data1["test_column"] = 1

    assert "test_column" not in mpdta_data2.columns
