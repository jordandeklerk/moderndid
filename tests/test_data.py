"""Test the datasets."""

from moderndid.core.data import load_ehec, load_mpdta, load_nsw
from tests.helpers import importorskip

pl = importorskip("polars")


def test_load_nsw():
    nsw_data = load_nsw()

    assert isinstance(nsw_data, pl.DataFrame)
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

    assert set(nsw_data["year"].unique().to_list()) == {1975, 1978}
    assert set(nsw_data["experimental"].unique().to_list()) == {0, 1}

    assert nsw_data["experimental"].sum() > 0
    assert (nsw_data["experimental"] == 0).sum() > 0


def test_load_nsw_data_integrity():
    nsw_data = load_nsw()

    key_columns = ["id", "year", "experimental"]
    for col in key_columns:
        assert nsw_data[col].is_not_null().all()

    assert nsw_data["id"].dtype.is_integer()
    assert nsw_data["year"].dtype.is_integer()
    assert nsw_data["experimental"].dtype.is_integer()
    assert nsw_data["re"].dtype.is_float()

    id_counts = nsw_data.group_by("id").len()
    assert (id_counts["len"] == 2).all()


def test_import_from_package():
    import moderndid

    assert hasattr(moderndid, "data")
    assert hasattr(moderndid.data, "load_nsw")

    nsw_data = moderndid.data.load_nsw()

    assert isinstance(nsw_data, pl.DataFrame)
    assert nsw_data.shape[0] > 0


def test_load_nsw_returns_copy():
    nsw_data1 = load_nsw()
    nsw_data2 = load_nsw()

    nsw_data1 = nsw_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in nsw_data2.columns


def test_load_mpdta():
    mpdta_data = load_mpdta()

    assert isinstance(mpdta_data, pl.DataFrame)
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

    assert set(mpdta_data["year"].unique().to_list()) == {2003, 2004, 2005, 2006, 2007}
    assert set(mpdta_data["treat"].unique().to_list()) == {0, 1}
    assert set(mpdta_data["first.treat"].unique().to_list()) == {0, 2004, 2006, 2007}


def test_load_mpdta_data_integrity():
    mpdta_data = load_mpdta()

    key_columns = ["year", "countyreal", "lpop", "lemp", "first.treat", "treat"]
    for col in key_columns:
        assert mpdta_data[col].is_not_null().all()

    assert mpdta_data["year"].dtype.is_integer()
    assert mpdta_data["countyreal"].dtype.is_integer()
    assert mpdta_data["lpop"].dtype.is_float()
    assert mpdta_data["lemp"].dtype.is_float()
    assert mpdta_data["first.treat"].dtype.is_integer()
    assert mpdta_data["treat"].dtype.is_integer()

    county_counts = mpdta_data.group_by("countyreal").len()
    assert (county_counts["len"] == 5).all()

    for county in mpdta_data["countyreal"].unique().to_list():
        county_data = mpdta_data.filter(pl.col("countyreal") == county)
        assert county_data["first.treat"].n_unique() == 1


def test_import_mpdta_from_package():
    import moderndid

    assert hasattr(moderndid, "load_mpdta")
    assert hasattr(moderndid.data, "load_mpdta")

    mpdta_data = moderndid.load_mpdta()

    assert isinstance(mpdta_data, pl.DataFrame)
    assert mpdta_data.shape[0] == 2500


def test_load_mpdta_returns_copy():
    mpdta_data1 = load_mpdta()
    mpdta_data2 = load_mpdta()

    mpdta_data1 = mpdta_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in mpdta_data2.columns


def test_load_ehec():
    ehec_data = load_ehec()

    assert isinstance(ehec_data, pl.DataFrame)
    assert ehec_data.shape == (552, 5)

    expected_columns = ["stfips", "year", "dins", "yexp2", "W"]
    assert list(ehec_data.columns) == expected_columns

    assert set(ehec_data["year"].unique().to_list()) == set(range(2008, 2020))

    assert ehec_data["yexp2"].is_not_null().sum() == 360
    assert ehec_data["yexp2"].is_null().sum() == 192

    assert set(ehec_data["yexp2"].drop_nulls().unique().to_list()) == {2014.0, 2015.0, 2016.0, 2017.0, 2019.0}


def test_load_ehec_data_integrity():
    ehec_data = load_ehec()

    key_columns = ["stfips", "year", "dins", "W"]
    for col in key_columns:
        assert ehec_data[col].is_not_null().all()

    assert ehec_data["stfips"].dtype.is_integer()
    assert ehec_data["year"].dtype.is_integer()
    assert ehec_data["dins"].dtype.is_float()
    assert ehec_data["yexp2"].dtype.is_float()
    assert ehec_data["W"].dtype.is_float()

    state_counts = ehec_data.group_by("stfips").len()
    assert (state_counts["len"] == 12).all()

    assert ehec_data["dins"].min() > 0.4
    assert ehec_data["dins"].max() < 1.0

    assert ehec_data["W"].min() > 0


def test_import_ehec_from_package():
    import moderndid

    assert hasattr(moderndid, "load_ehec")
    assert hasattr(moderndid.data, "load_ehec")

    ehec_data = moderndid.load_ehec()

    assert isinstance(ehec_data, pl.DataFrame)
    assert ehec_data.shape[0] == 552


def test_load_ehec_returns_copy():
    ehec_data1 = load_ehec()
    ehec_data2 = load_ehec()

    ehec_data1 = ehec_data1.with_columns(pl.lit(1).alias("test_column"))

    assert "test_column" not in ehec_data2.columns
