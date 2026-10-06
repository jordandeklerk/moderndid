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
def etwfe_poisson_id(mpdta_data):
    return etwfe(
        data=mpdta_data,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        family="poisson",
    )


@pytest.fixture
def mpdta_moderators(mpdta_data):
    gls_states = [17, 18, 26, 27, 36, 39, 42, 55]
    lpop = pl.col("lpop")
    popcat = (
        pl.when(lpop <= lpop.quantile(1 / 3))
        .then(pl.lit("low"))
        .when(lpop <= lpop.quantile(2 / 3))
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
    )
    return mpdta_data.with_columns((pl.col("countyreal") // 1000).is_in(gls_states).alias("gls")).with_columns(
        (~pl.col("gls")).alias("notgls"),
        pl.col("gls").cast(pl.Float64).alias("gls01"),
        (2 * pl.col("lpop") + 3).alias("lpop_aff"),
        (pl.col("lemp") > pl.col("lemp").median()).cast(pl.Int64).alias("ybin"),
        popcat.alias("popcat"),
        popcat.replace("high", "z_high").alias("popcat2"),
    )


@pytest.fixture
def mpdta_unbalanced_moderators(mpdta_moderators):
    trend = 0.05 * (pl.col("countyreal") % 5 - 2) * (pl.col("year") - 2005)
    data = mpdta_moderators.with_columns((pl.col("lpop") + trend).alias("x_tv"))
    return data.filter(~((pl.col("countyreal") % 7 == 0) & (pl.col("year") == 2005)))


@pytest.fixture
def mpdta_state_dummies(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))
    states = sorted(data["state"].unique().to_list())[1:]
    data = data.with_columns([(pl.col("state") == s).cast(pl.Float64).alias(f"st{s}") for s in states])
    return data, "~" + " + ".join(f"st{s}" for s in states)


@pytest.fixture
def mpdta_renamed(mpdta_data):
    lpop = pl.col("lpop")
    popcat = (
        pl.when(lpop <= lpop.quantile(1 / 3))
        .then(pl.lit("low"))
        .when(lpop <= lpop.quantile(2 / 3))
        .then(pl.lit("mid"))
        .otherwise(pl.lit("high"))
    )
    weights = 1 + (pl.col("countyreal") % 7) / 7
    return mpdta_data.with_columns(
        pl.col("first.treat").alias("first_treat"),
        pl.col("first.treat").alias("first treat"),
        pl.col("countyreal").alias("county.id"),
        pl.col("countyreal").alias("county id"),
        pl.col("year").alias("year.t"),
        pl.col("year").alias("year t"),
        pl.col("lpop").alias("log.pop"),
        pl.col("lpop").alias("log pop"),
        pl.col("lemp").alias("l.emp"),
        pl.col("lemp").alias("l emp"),
        pl.col("lemp").alias("l-emp"),
        weights.alias("w"),
        weights.alias("w t"),
        popcat.alias("popcat"),
        (popcat + pl.lit(" pop")).alias("popcat spaced"),
    )


@pytest.fixture
def mpdta_never_codes(mpdta_data):
    never = pl.col("first.treat") == 0
    cohort = pl.col("first.treat").cast(pl.Float64)
    codes = {"null": None, "nan": float("nan"), "inf": float("inf"), "9999": 9999.0}
    return {
        label: mpdta_data.with_columns(
            pl.when(never).then(pl.lit(code, dtype=pl.Float64)).otherwise(cohort).alias("first.treat")
        )
        for label, code in codes.items()
    }


@pytest.fixture
def mpdta_late_cohort(mpdta_data):
    late = pl.when(pl.col("first.treat") == 2006)
    return (
        mpdta_data.with_columns(late.then(2008).otherwise(pl.col("first.treat")).alias("first.treat")),
        mpdta_data.with_columns(late.then(0).otherwise(pl.col("first.treat")).alias("first.treat")),
    )


@pytest.fixture
def mpdta_always_treated(mpdta_data):
    ids = mpdta_data.filter(pl.col("first.treat") == 0)["countyreal"].unique().sort().tail(30)
    in_ids = pl.col("countyreal").is_in(ids.implode())
    return (
        mpdta_data.with_columns(pl.when(in_ids).then(2003).otherwise(pl.col("first.treat")).alias("first.treat")),
        mpdta_data.filter(~in_ids),
    )


@pytest.fixture
def mpdta_missing(mpdta_data):
    data = mpdta_data.with_columns(
        (1 + (pl.col("countyreal") % 7) / 7).alias("w"),
        (pl.col("countyreal") // 1000).is_in([17, 18, 26, 27, 36, 39, 42, 55]).alias("gls"),
    )
    return data, (pl.col("countyreal") * 31 + pl.col("year")) % 17 == 0


@pytest.fixture
def mpdta_singletons(mpdta_data):
    never = mpdta_data.filter(pl.col("first.treat") == 0)["countyreal"].unique().sort().head(15)
    early = mpdta_data.filter(pl.col("first.treat") == 2004)["countyreal"].unique().sort().head(10)
    single = pl.col("countyreal").is_in(pl.concat([never, early]).implode())
    return mpdta_data.filter(~single | (pl.col("year") == 2006))


@pytest.fixture
def mpdta_no_never(mpdta_data):
    return mpdta_data.filter(pl.col("first.treat") != 0)


@pytest.fixture
def mpdta_calendar_gap(mpdta_data):
    return mpdta_data.filter(pl.col("year") != 2005)


@pytest.fixture
def mpdta_late_entry(mpdta_data):
    return mpdta_data.filter(~((pl.col("first.treat") == 2006) & (pl.col("year") < 2006)))


@pytest.fixture
def mpdta_states(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") // 1000).cast(pl.Float64).alias("st"))
    return data, pl.col("countyreal") % 50 == 0


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


@pytest.fixture(params=["repeated_row", "hidden_gap", "relabeled_row"])
def mpdta_duplicated(request, mpdta_data):
    """mpdta in which county 17005 has two rows in 2005."""
    county = pl.col("countyreal") == 17005
    row = mpdta_data.filter(county & (pl.col("year") == 2005))
    if request.param == "repeated_row":
        return pl.concat([mpdta_data, row])
    if request.param == "hidden_gap":
        return pl.concat([mpdta_data.filter(~(county & (pl.col("year") == 2006))), row])
    return pl.concat([mpdta_data, row.with_columns(pl.col("lemp") + 1)])


@pytest.fixture
def mpdta_without_years(mpdta_data):
    """mpdta plus two rows of county 17005 whose year is missing."""
    rows = mpdta_data.filter((pl.col("countyreal") == 17005) & (pl.col("year") == 2005))
    return pl.concat([mpdta_data, pl.concat([rows, rows]).with_columns(pl.lit(None, dtype=pl.Int64).alias("year"))])
