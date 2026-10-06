import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import load_mpdta


@pytest.fixture
def mpdta_data():
    return load_mpdta()


@pytest.fixture
def mpdta_unbalanced(mpdta_data):
    return mpdta_data.filter(~((pl.col("countyreal") == 8001) & (pl.col("year") == 2007)))


@pytest.fixture
def mpdta_without_county_8001(mpdta_data):
    return mpdta_data.filter(pl.col("countyreal") != 8001)


@pytest.fixture
def mpdta_without_never_treated(mpdta_data):
    return mpdta_data.filter(pl.col("first.treat") != 0)


@pytest.fixture
def mpdta_spec():
    return {
        "yname": "lemp",
        "tname": "year",
        "idname": "countyreal",
        "gname": "first.treat",
        "xformla": "~ lpop",
    }


@pytest.fixture
def didml_options():
    return {"k_folds": 2, "random_state": 0}


@pytest.fixture
def mpdta_repeated_row(mpdta_data):
    return pl.concat([mpdta_data, mpdta_data.filter((pl.col("countyreal") == 17005) & (pl.col("year") == 2005))])


@pytest.fixture
def mpdta_infinite_outcome(mpdta_data):
    row = (pl.col("countyreal") == 8001) & (pl.col("year") == 2007)
    return mpdta_data.with_columns(pl.when(row).then(float("inf")).otherwise(pl.col("lemp")).alias("lemp"))


@pytest.fixture
def mpdta_zero_weights(mpdta_data):
    return mpdta_data.with_columns(pl.lit(0.0).alias("w"))
