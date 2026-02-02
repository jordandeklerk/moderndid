import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import att_gt, load_mpdta


@pytest.fixture
def mpdta_data():
    return load_mpdta()


@pytest.fixture
def att_gt_baseline_result(mpdta_data):
    return att_gt(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        boot=False,
    )


@pytest.fixture
def mpdta_converted(request, mpdta_data):
    df_type = request.param
    if df_type == "pandas":
        importorskip("pandas")
        return mpdta_data.to_pandas()
    if df_type == "pyarrow":
        importorskip("pyarrow")
        return mpdta_data.to_arrow()
    if df_type == "duckdb":
        duckdb = importorskip("duckdb")
        conn = duckdb.connect()
        conn.register("mpdta", mpdta_data.to_arrow())
        return conn.execute("SELECT * FROM mpdta").fetch_arrow_table()
    raise ValueError(f"Unknown dataframe type: {df_type}")
