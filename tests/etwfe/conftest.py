import pytest

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("pyfixest")

from moderndid import etwfe, load_mpdta
from moderndid.core.preprocess.config import EtwfeConfig


@pytest.fixture
def mpdta_data():
    return load_mpdta()


@pytest.fixture
def base_config():
    return EtwfeConfig(
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~1",
        cgroup="notyet",
        fe="vs",
        alp=0.05,
        panel=True,
    )


@pytest.fixture
def etwfe_baseline(mpdta_data):
    return etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
    )


@pytest.fixture
def etwfe_never(mpdta_data):
    return etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        cgroup="never",
    )


@pytest.fixture
def etwfe_with_covariates(mpdta_data):
    return etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ lpop",
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
