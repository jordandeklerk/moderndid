import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import att_gt, load_mpdta


@pytest.fixture
def mpdta_data():
    return load_mpdta()


@pytest.fixture
def mpdta_pop_weighted(mpdta_data):
    return mpdta_data.with_columns(pl.col("lpop").exp().alias("pop"))


@pytest.fixture
def mpdta_unbalanced(mpdta_data):
    return mpdta_data.filter(~((pl.col("countyreal") % 7 == 0) & (pl.col("year") == 2005)))


@pytest.fixture
def mpdta_rotating(mpdta_data):
    return mpdta_data.filter((pl.col("countyreal") + pl.col("year")) % 5 < 3)


@pytest.fixture
def mpdta_unbalanced_varying_weights(mpdta_unbalanced):
    return mpdta_unbalanced.with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 4)).alias("w")
    )


@pytest.fixture
def mpdta_varying_weights(mpdta_data):
    return mpdta_data.with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 4)).alias("w")
    )


@pytest.fixture
def mpdta_weights_by_year(mpdta_varying_weights):
    return {
        year: mpdta_varying_weights.join(
            mpdta_varying_weights.filter(pl.col("year") == year).select("countyreal", pl.col("w").alias("w_fixed")),
            on="countyreal",
        )
        for year in (2003, 2004, 2005, 2006, 2007)
    }


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


@pytest.fixture
def mpdta_one_nan(request, mpdta_data):
    column = request.param
    data = mpdta_data.with_columns(pl.col("lpop").exp().alias("pop"), (pl.col("countyreal") % 10).alias("cluster"))
    county = data.filter(pl.col("first.treat") == 2004)["countyreal"].min()
    row = (pl.col("countyreal") == county) & (pl.col("year") == 2005)
    return data.with_columns(pl.when(row).then(float("nan")).otherwise(pl.col(column).cast(pl.Float64)).alias(column))


@pytest.fixture
def mpdta_one_infinite(request, mpdta_data):
    column, value = request.param
    data = mpdta_data.with_columns(pl.col("lpop").exp().alias("pop"), (pl.col("countyreal") % 10).alias("cluster"))
    county = data.filter(pl.col("first.treat") == 2004)["countyreal"].min()
    row = (pl.col("countyreal") == county) & (pl.col("year") == 2005)
    return data.with_columns(pl.when(row).then(value).otherwise(pl.col(column).cast(pl.Float64)).alias(column))


@pytest.fixture
def mpdta_bad_weights(request, mpdta_data):
    county = mpdta_data.filter(pl.col("first.treat") == 2004)["countyreal"].min()
    row = (pl.col("countyreal") == county) & (pl.col("year") == 2005)
    weights = {
        "zero": pl.lit(0.0),
        "zero outside an infinite row": pl.when(row).then(float("inf")).otherwise(0.0),
        "one negative": pl.when(row).then(-1.0).otherwise(pl.col("lpop").exp()),
    }
    return mpdta_data.with_columns(weights[request.param].alias("w"))


@pytest.fixture
def mpdta_negative_weight_missing_outcome(mpdta_pop_weighted):
    county = mpdta_pop_weighted.filter(pl.col("first.treat") == 2004)["countyreal"].min()
    row = (pl.col("countyreal") == county) & (pl.col("year") == 2005)
    return mpdta_pop_weighted.with_columns(
        pl.when(row).then(-1.0).otherwise(pl.col("pop")).alias("pop"),
        pl.when(row).then(None).otherwise(pl.col("lemp")).alias("lemp"),
    )


@pytest.fixture
def mpdta_nan_cohort(mpdta_data):
    counties = mpdta_data.filter(pl.col("first.treat") == 2007)["countyreal"].unique().sort().head(10)
    return mpdta_data.with_columns(
        pl.when(pl.col("countyreal").is_in(counties.implode()))
        .then(float("nan"))
        .otherwise(pl.col("first.treat").cast(pl.Float64))
        .alias("first.treat")
    )


@pytest.fixture
def mpdta_negative_never_treated(mpdta_data):
    return mpdta_data.with_columns(
        pl.when(pl.col("first.treat") == 0).then(-1).otherwise(pl.col("first.treat")).alias("first.treat")
    )


@pytest.fixture
def mpdta_infinite_never_treated(mpdta_data):
    return mpdta_data.with_columns(
        pl.when(pl.col("first.treat") == 0).then(float("inf")).otherwise(pl.col("first.treat")).alias("first.treat")
    )


@pytest.fixture
def mpdta_without_never_treated(mpdta_data):
    return mpdta_data.filter(pl.col("first.treat") != 0)


@pytest.fixture
def mpdta_cohort_after_panel(mpdta_data):
    return mpdta_data.with_columns(
        pl.when(pl.col("first.treat") == 2007).then(2008).otherwise(pl.col("first.treat")).alias("first.treat")
    )


@pytest.fixture
def mpdta_early_cohorts(mpdta_data):
    """mpdta in which counties below 9000 start treatment in 2003 and the others below 13000 in 2004."""
    return mpdta_data.with_columns(
        pl.when(pl.col("countyreal") < 9000)
        .then(2003)
        .when(pl.col("countyreal") < 13000)
        .then(2004)
        .otherwise(pl.col("first.treat"))
        .alias("first.treat")
    )


@pytest.fixture
def cohort_codes_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 3, 1, 3, 1, 3],
            "y": [0.1, 0.4, 0.2, 0.9, 0.3, 0.5],
            "g": [0, 0, 4, 4, 5, 5],
        }
    )


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
def mpdta_shuffled(mpdta_data):
    return mpdta_data.sample(fraction=1.0, shuffle=True, seed=7)


@pytest.fixture
def unit_period_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "time": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "y": [0.1, 0.4, 0.2, 0.9, 0.3, 0.5, 0.6, 1.1, 0.2, 0.8],
            "g": [0, 0, 2, 2, 0, 0, 2, 2, 0, 0],
        }
    )
