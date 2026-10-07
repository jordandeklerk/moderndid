"""Shared fixtures for validation tests."""

import numpy as np
import polars as pl
import pytest

from moderndid import (
    ddd_mp,
    gen_ddd_mult_periods,
    gen_ddd_scalable,
    load_cai2016,
    load_engel,
    load_favara_imbs,
    load_mpdta,
    load_nsw,
)
from moderndid.didtriple.dgp import gen_ddd_2periods


@pytest.fixture
def two_period_dgp_result():
    """Full 2-period DGP result including true ATT and oracle ATT."""
    result = gen_ddd_2periods(n=1000, dgp_type=1, random_state=42)
    return result["data"], result["true_att"], result["oracle_att"]


@pytest.fixture
def mp_ddd_data():
    """Generate multi-period panel data for DDD testing."""
    rng = np.random.default_rng(42)
    n_units = 500
    time_periods = [1, 2, 3, 4, 5]

    unit_ids = np.arange(n_units)
    groups = rng.choice([0, 3, 4], size=n_units, p=[0.5, 0.25, 0.25])
    partition = rng.choice([0, 1], size=n_units, p=[0.5, 0.5])

    records = []
    for unit in unit_ids:
        g = groups[unit]
        p = partition[unit]
        unit_effect = rng.normal(0, 1)

        for t in time_periods:
            time_effect = 0.5 * t
            treat_effect = 0.0
            if 0 < g <= t and p == 1:
                treat_effect = 2.0

            y = unit_effect + time_effect + treat_effect + rng.normal(0, 0.5)
            records.append({"id": unit, "time": t, "y": y, "group": g, "partition": p})

    return pl.DataFrame(records)


@pytest.fixture
def mp_ddd_unbalanced_data(mp_ddd_data):
    """Drop about 8 percent of the multi-period panel rows to unbalance it."""
    rng = np.random.default_rng(7)
    return mp_ddd_data.filter(pl.Series(rng.random(len(mp_ddd_data)) >= 0.08))


@pytest.fixture
def two_period_clustered_data(two_period_dgp_result):
    """Nest clusters of unequal size in the states and shock the eligible units of each cluster in period 2."""
    data, _, _ = two_period_dgp_result
    data = data.with_columns((100 * pl.col("state") + pl.col("id").sqrt().floor()).cast(pl.Int64).alias("cluster"))
    clusters = data["cluster"].unique().sort().to_list()
    shocks = dict(zip(clusters, np.random.default_rng(7).normal(0, 4, len(clusters))))
    shock = pl.col("cluster").replace_strict(shocks, return_dtype=pl.Float64)
    return data.with_columns(pl.col("y") + (pl.col("time") == 2).cast(pl.Float64) * pl.col("partition") * shock)


@pytest.fixture
def two_period_rcs_clustered_data(two_period_rcs_data):
    """Nest clusters of unequal size in the states and shock the eligible observations of each cluster in period 1."""
    data = two_period_rcs_data.with_columns(
        (100 * pl.col("state") + (pl.col("id") % 1000 + 1).sqrt().floor()).cast(pl.Int64).alias("cluster")
    )
    clusters = data["cluster"].unique().sort().to_list()
    shocks = dict(zip(clusters, np.random.default_rng(8).normal(0, 1, len(clusters))))
    shock = pl.col("cluster").replace_strict(shocks, return_dtype=pl.Float64)
    return data.with_columns(pl.col("y") + (pl.col("time") == 1).cast(pl.Float64) * pl.col("partition") * shock)


@pytest.fixture
def mp_ddd_clustered_data(mp_ddd_data):
    """Nest clusters of unequal size in the groups and shock the eligible units of each cluster from period 3 on."""
    data = mp_ddd_data.with_columns(
        (100 * pl.col("group") + (pl.col("id") + 1).sqrt().floor()).cast(pl.Int64).alias("cluster")
    )
    clusters = data["cluster"].unique().sort().to_list()
    shocks = dict(zip(clusters, np.random.default_rng(9).normal(0, 1, len(clusters))))
    shock = pl.col("cluster").replace_strict(shocks, return_dtype=pl.Float64)
    return data.with_columns(pl.col("y") + (pl.col("time") >= 3).cast(pl.Float64) * pl.col("partition") * shock)


@pytest.fixture(scope="module")
def mpdta_pop_weighted():
    """Weight the mpdta counties by population in a column named pop."""
    return load_mpdta().with_columns(pl.col("lpop").exp().alias("pop"))


@pytest.fixture(scope="module")
def mpdta_pop_weighted_csv_path(mpdta_pop_weighted, tmp_path_factory):
    """Write the population-weighted mpdta panel to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_pop_weighted.csv"
    mpdta_pop_weighted.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_repeated_row():
    """Append a second copy of the 2005 record of county 17005 to mpdta."""
    data = load_mpdta()
    return pl.concat([data, data.filter((pl.col("countyreal") == 17005) & (pl.col("year") == 2005))])


@pytest.fixture(scope="module")
def mpdta_repeated_row_csv_path(mpdta_repeated_row, tmp_path_factory):
    """Write the mpdta panel with the repeated record to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_repeated_row.csv"
    mpdta_repeated_row.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_without_years():
    """Append two records of county 17005 whose year is missing to mpdta."""
    data = load_mpdta()
    rows = data.filter((pl.col("countyreal") == 17005) & (pl.col("year") == 2005))
    return pl.concat([data, pl.concat([rows, rows]).with_columns(pl.lit(None, dtype=pl.Int64).alias("year"))])


@pytest.fixture(scope="module")
def mpdta_without_years_csv_path(mpdta_without_years, tmp_path_factory):
    """Write the mpdta panel with the yearless records to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_without_years.csv"
    mpdta_without_years.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def nsw_repeated_row():
    """Append a second copy of the 1975 record of unit 15995 to the NSW panel."""
    data = load_nsw().select("id", "year", "re", "experimental")
    return pl.concat([data, data.filter((pl.col("id") == 15995) & (pl.col("year") == 1975))])


@pytest.fixture(scope="module")
def nsw_repeated_row_csv_path(nsw_repeated_row, tmp_path_factory):
    """Write the NSW panel with the repeated record to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("nsw") / "nsw_repeated_row.csv"
    nsw_repeated_row.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_unbalanced():
    """Drop the 2005 record of each mpdta county whose id is a multiple of seven."""
    return load_mpdta().filter(~((pl.col("countyreal") % 7 == 0) & (pl.col("year") == 2005)))


@pytest.fixture(scope="module")
def mpdta_unbalanced_csv_path(mpdta_unbalanced, tmp_path_factory):
    """Write the unbalanced mpdta panel to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_unbalanced.csv"
    mpdta_unbalanced.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_unbalanced_varying_weights(mpdta_unbalanced):
    """Weight the unbalanced mpdta panel by population times a factor that changes from year to year."""
    return mpdta_unbalanced.with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 4)).alias("w")
    )


@pytest.fixture(scope="module")
def mpdta_unbalanced_varying_weights_csv_path(mpdta_unbalanced_varying_weights, tmp_path_factory):
    """Write the unbalanced mpdta panel with time-varying weights to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_unbalanced_varying_weights.csv"
    mpdta_unbalanced_varying_weights.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_varying_weights():
    """Weight the balanced mpdta panel by population times a factor that changes from year to year."""
    return load_mpdta().with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 4)).alias("w")
    )


@pytest.fixture(scope="module")
def mpdta_varying_weights_csv_path(mpdta_varying_weights, tmp_path_factory):
    """Write the balanced mpdta panel with time-varying weights to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_varying_weights.csv"
    mpdta_varying_weights.write_csv(path)
    return str(path)


@pytest.fixture
def mpdta_one_infinite(request):
    """Weight mpdta by population in pop and put an infinite value in the 2005 record of county 17005."""
    column, value = request.param
    row = (pl.col("countyreal") == 17005) & (pl.col("year") == 2005)
    data = load_mpdta().with_columns(pl.col("lpop").exp().alias("pop"))
    return data.with_columns(pl.when(row).then(value).otherwise(pl.col(column).cast(pl.Float64)).alias(column))


@pytest.fixture
def mpdta_one_infinite_csv_path(mpdta_one_infinite, tmp_path):
    """Write the mpdta panel with the infinite value to a CSV file and return its path."""
    path = tmp_path / "mpdta_one_infinite.csv"
    mpdta_one_infinite.write_csv(path)
    return str(path)


@pytest.fixture
def mpdta_one_missing(request):
    """Cluster mpdta by the last digit of the county and leave one value missing in the 2005 record of 17005."""
    column, value = request.param
    row = (pl.col("countyreal") == 17005) & (pl.col("year") == 2005)
    data = load_mpdta().with_columns((pl.col("countyreal") % 10).alias("cluster"))
    return data.with_columns(pl.when(row).then(value).otherwise(pl.col(column).cast(pl.Float64)).alias(column))


@pytest.fixture
def mpdta_one_missing_csv_path(mpdta_one_missing, tmp_path):
    """Write the mpdta panel with the missing value to a CSV file and return its path."""
    path = tmp_path / "mpdta_one_missing.csv"
    mpdta_one_missing.write_csv(path)
    return str(path)


@pytest.fixture
def mpdta_bad_weights(request):
    """Give mpdta weights in w that are all zero, zero outside one infinite record, or negative in one record."""
    row = (pl.col("countyreal") == 17005) & (pl.col("year") == 2005)
    weights = {
        "zero": pl.lit(0.0),
        "zero outside an infinite row": pl.when(row).then(float("inf")).otherwise(0.0),
        "one negative": pl.when(row).then(-1.0).otherwise(pl.col("lpop").exp()),
    }
    return load_mpdta().with_columns(weights[request.param].alias("w"))


@pytest.fixture
def mpdta_bad_weights_csv_path(mpdta_bad_weights, tmp_path):
    """Write the mpdta panel with the invalid weights to a CSV file and return its path."""
    path = tmp_path / "mpdta_bad_weights.csv"
    mpdta_bad_weights.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_unbalanced_clustered(mpdta_unbalanced):
    """Add a cluster column that holds the last digit of the county id to the unbalanced mpdta panel."""
    return mpdta_unbalanced.with_columns((pl.col("countyreal") % 10).alias("cluster"))


@pytest.fixture(scope="module")
def mpdta_unbalanced_clustered_csv_path(mpdta_unbalanced_clustered, tmp_path_factory):
    """Write the unbalanced mpdta panel with its cluster column to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_unbalanced_clustered.csv"
    mpdta_unbalanced_clustered.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_rotating():
    """Keep each mpdta county in three of the five years, the way a rotating survey samples households."""
    return load_mpdta().filter((pl.col("countyreal") + pl.col("year")) % 5 < 3)


@pytest.fixture(scope="module")
def mpdta_rotating_csv_path(mpdta_rotating, tmp_path_factory):
    """Write the rotating mpdta sample to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_rotating.csv"
    mpdta_rotating.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def mpdta_formula_columns():
    """Add a dotted copy of lpop and the columns that the terms I(lpop^2) and log(lpop) build."""
    return load_mpdta().with_columns(
        pl.col("lpop").alias("log.pop"),
        (pl.col("lpop") ** 2).alias("lpop_sq"),
        pl.col("lpop").log().alias("log_lpop"),
    )


@pytest.fixture(scope="module")
def mpdta_formula_columns_csv_path(mpdta_formula_columns, tmp_path_factory):
    """Write mpdta with the formula columns to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_formula_columns.csv"
    mpdta_formula_columns.write_csv(path)
    return str(path)


@pytest.fixture
def cai_data():
    """Load the Cai (2016) households with group 2003 for the treated regions."""
    return load_cai2016().with_columns((pl.col("treatment") * 2003).alias("group"))


@pytest.fixture
def cai_balanced_data(cai_data):
    """Keep the Cai (2016) households observed in all nine years."""
    return cai_data.filter(pl.len().over("hhno") == 9)


@pytest.fixture
def mp_ddd_result(mp_ddd_data):
    """Get multi-period DDD result for aggregation tests."""
    return ddd_mp(
        data=mp_ddd_data,
        y_col="y",
        time_col="time",
        id_col="id",
        group_col="group",
        partition_col="partition",
        est_method="reg",
    )


@pytest.fixture
def two_period_rcs_data():
    """Generate 2-period repeated cross-section data for DDD testing."""
    rng = np.random.default_rng(42)

    n_per_period = 1000
    records = []

    for t in [0, 1]:
        state = rng.choice([0, 1], size=n_per_period, p=[0.5, 0.5])
        partition = rng.choice([0, 1], size=n_per_period, p=[0.5, 0.5])

        for i in range(n_per_period):
            s = state[i]
            p = partition[i]

            cov1 = rng.normal(0, 1)
            cov2 = rng.normal(0, 1)
            cov3 = rng.normal(0, 1)
            cov4 = rng.normal(0, 1)

            base_y = 1.0 + 0.5 * cov1 + 0.3 * cov2 + 0.2 * cov3 + 0.1 * cov4
            time_effect = 0.5 * t
            treat_effect = 0.0
            if s == 1 and p == 1 and t == 1:
                treat_effect = 2.0

            y = base_y + time_effect + treat_effect + rng.normal(0, 0.5)

            records.append(
                {
                    "id": len(records),
                    "time": t,
                    "y": y,
                    "state": s,
                    "partition": p,
                    "cov1": cov1,
                    "cov2": cov2,
                    "cov3": cov3,
                    "cov4": cov4,
                }
            )

    return pl.DataFrame(records)


@pytest.fixture
def mp_rcs_data():
    """Generate multi-period repeated cross-section data for DDD testing."""
    rng = np.random.default_rng(42)

    n_per_period = 300
    time_periods = [1, 2, 3, 4, 5]
    records = []

    for t in time_periods:
        groups = rng.choice([0, 3, 4], size=n_per_period, p=[0.5, 0.25, 0.25])
        partition = rng.choice([0, 1], size=n_per_period, p=[0.5, 0.5])

        for i in range(n_per_period):
            g = groups[i]
            p = partition[i]

            base_y = rng.normal(0, 1)
            time_effect = 0.5 * t
            treat_effect = 0.0
            if 0 < g <= t and p == 1:
                treat_effect = 2.0

            y = base_y + time_effect + treat_effect + rng.normal(0, 0.5)

            records.append(
                {
                    "id": len(records),
                    "time": t,
                    "y": y,
                    "group": g,
                    "partition": p,
                }
            )

    return pl.DataFrame(records)


@pytest.fixture
def engel_data():
    """Load Engel dataset for NPIV validation tests."""
    engel_df = load_engel()
    engel_df = engel_df.sort("logexp")

    return {
        "food": engel_df["food"].to_numpy(),
        "logexp": engel_df["logexp"].to_numpy().reshape(-1, 1),
        "logwages": engel_df["logwages"].to_numpy().reshape(-1, 1),
    }


@pytest.fixture(scope="module")
def two_regressor_iv_data():
    """Two endogenous regressors, two instruments, and an evaluation grid for NPIV validation tests."""
    rng = np.random.default_rng(3)
    n = 600
    w = rng.uniform(size=(n, 2))
    v = rng.uniform(size=(n, 2))
    x = 0.7 * w + 0.3 * v
    y = np.sin(2 * x[:, 0]) + x[:, 1] ** 2 + 0.3 * (v[:, 0] - 0.5) + 0.2 * rng.normal(size=n)
    x_eval = np.column_stack([np.linspace(0.2, 0.8, 30), np.linspace(0.3, 0.7, 30)])
    return y, x, w, x_eval


@pytest.fixture(scope="module")
def didinter_unbalanced_data():
    """Clustered panel with missing outcomes, missing rows, and doses that keep rising after the switch."""
    rng = np.random.default_rng(20261002)
    n_periods = 8
    rows = []
    unit = 0
    for baseline, n_units, kind in [(0, 90, None), (0, 150, "step"), (0, 50, "jump"), (1, 35, None), (1, 30, "raise")]:
        for _ in range(n_units):
            unit += 1
            effect = rng.normal()
            path = [baseline] * n_periods
            if kind == "step":
                first = int(rng.choice([3, 4, 5, 6, 7]))
                second = first + int(rng.choice([1, 2])) if rng.random() < 0.5 else None
                for t in range(1, n_periods + 1):
                    if t >= first:
                        path[t - 1] = 1
                    if second is not None and t >= second:
                        path[t - 1] = 2
            elif kind == "jump":
                first = int(rng.choice([3, 4, 5, 6]))
                for t in range(first, n_periods + 1):
                    path[t - 1] = 2
            elif kind == "raise":
                first = int(rng.choice([4, 5, 6]))
                for t in range(first, n_periods + 1):
                    path[t - 1] = 2
            weight = float(rng.uniform(0.5, 2.0))
            for t in range(1, n_periods + 1):
                dose = path[t - 1]
                lagged = path[t - 2] if t > 1 else baseline
                y = effect + 0.3 * t + (dose - baseline) + 0.5 * (lagged - baseline) + rng.normal()
                rows.append((unit, t, float(dose), y, weight, (unit - 1) // 14 + 1))

    df = pl.DataFrame(rows, schema=["g", "t", "d", "y", "w", "cl"], orient="row")
    missing_y = rng.random(df.height) < 0.06
    df = df.with_columns(pl.when(pl.Series(missing_y)).then(None).otherwise(pl.col("y")).alias("y"))
    dropped = (rng.random(df.height) < 0.03) & (df["t"] > 1).to_numpy()
    first_dropped = np.isin(df["g"].to_numpy(), [5, 160, 260]) & (df["t"] == 1).to_numpy()
    return df.filter(~pl.Series(dropped | first_dropped))


@pytest.fixture(scope="module")
def didinter_unbalanced_csv_path(didinter_unbalanced_data, tmp_path_factory):
    """Write the unbalanced clustered panel to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_unbalanced.csv"
    didinter_unbalanced_data.write_csv(path, null_value="NA")
    return str(path)


@pytest.fixture(scope="module")
def didinter_two_way_data():
    """Panel in which groups raise or lower their baseline treatment and some cohorts have a single group."""
    rng = np.random.default_rng(20261004)
    n_periods = 7
    rows = []
    unit = 0
    for baseline, step, n_units, dates in [
        (0, 0, 25, None),
        (0, 1, 30, [3, 4, 5, 6]),
        (1, 0, 20, None),
        (1, 1, 25, [3, 4, 5, 6]),
        (1, -1, 25, [3, 4, 5, 6]),
        (2, 0, 15, None),
        (2, -1, 20, [4, 5, 6]),
        (2, 1, 1, [5]),
        (2, -2, 1, [6]),
    ]:
        for _ in range(n_units):
            switch = int(rng.choice(dates)) if dates else n_periods + 1
            effect = rng.normal()
            weight = float(rng.uniform(0.5, 2.0))
            for t in range(1, n_periods + 1):
                dose = baseline + step * (t >= switch)
                lagged = baseline + step * (t - 1 >= switch)
                y = effect + 0.3 * t + (dose - baseline) + 0.5 * (lagged - baseline) + rng.normal()
                rows.append((unit + 1, t, float(dose), y, weight, unit // 9 + 1))
            unit += 1
    return pl.DataFrame(rows, schema=["g", "t", "d", "y", "w", "cl"], orient="row")


@pytest.fixture(scope="module")
def didinter_two_way_csv_path(didinter_two_way_data, tmp_path_factory):
    """Write the panel with treatment rises and falls to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_two_way.csv"
    didinter_two_way_data.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def didinter_zero_weight_clusters(didinter_two_way_data):
    """Panel with treatment rises and falls whose clusters 1 and 2 carry zero weight in every row."""
    return didinter_two_way_data.with_columns(pl.when(pl.col("cl") <= 2).then(0.0).otherwise(pl.col("w")).alias("w"))


@pytest.fixture(scope="module")
def didinter_zero_weight_clusters_csv_path(didinter_zero_weight_clusters, tmp_path_factory):
    """Write the panel with zero-weight clusters to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_zero_weight_clusters.csv"
    didinter_zero_weight_clusters.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def didinter_all_switch_data():
    """Clustered staggered panel in which every group eventually raises its treatment from 0 or lowers it from 1."""
    rng = np.random.default_rng(11)
    rows = []
    for unit in range(1, 201):
        baseline = float(unit > 120)
        switch = int(rng.integers(3, 7 if baseline else 9))
        effect = rng.normal()
        for t in range(1, 9):
            dose = 1 - baseline if t >= switch else baseline
            rows.append((unit, t, dose, effect + 0.2 * t + dose + rng.normal(scale=0.5), (unit - 1) // 10 + 1))
    return pl.DataFrame(rows, schema=["g", "t", "d", "y", "cl"], orient="row")


@pytest.fixture(scope="module")
def didinter_all_switch_csv_path(didinter_all_switch_data, tmp_path_factory):
    """Write the panel in which every group switches to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_all_switch.csv"
    didinter_all_switch_data.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def favara_gapped_years():
    """Favara and Imbs data with the years recoded to unevenly spaced numbers."""
    data = load_favara_imbs()
    first = data["year"].min()
    years = {year: year + (year - first) ** 2 for year in data["year"].unique().to_list()}
    return data.with_columns(pl.col("year").replace_strict(years))


@pytest.fixture(scope="module")
def favara_gapped_years_csv_path(favara_gapped_years, tmp_path_factory):
    """Write the Favara and Imbs data with unevenly spaced years to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "favara_gapped_years.csv"
    favara_gapped_years.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def favara_census_regions():
    """Favara and Imbs data with the Census region of each state."""
    regions = {
        1: [9, 23, 25, 33, 34, 36, 42, 44, 50],
        2: [17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55],
        3: [1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54],
        4: [2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56],
    }
    state_region = {state: region for region, states in regions.items() for state in states}
    return load_favara_imbs().with_columns(pl.col("state_n").replace_strict(state_region).alias("region"))


@pytest.fixture(scope="module")
def favara_census_regions_csv_path(favara_census_regions, tmp_path_factory):
    """Write the Favara and Imbs data with Census regions to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "favara_census_regions.csv"
    favara_census_regions.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def didinter_continuous_data():
    """Clustered panel with continuous baselines, a few of them shared, and treatment rises and falls."""
    rng = np.random.default_rng(20261005)
    rows = []
    for unit in range(1, 241):
        baseline = round(float(rng.uniform(0.5, 2.5)), 1 if unit % 10 == 0 else 6)
        switch = int(rng.integers(3, 7)) if rng.random() < 0.7 else None
        step = float(rng.choice([-1, 1]) * rng.uniform(0.5, 1.0))
        effect = rng.normal(1.0, 0.3)
        weight = float(rng.uniform(0.5, 2.0))
        for t in range(1, 8):
            dose = baseline + step if switch is not None and t >= switch else baseline
            x = rng.normal() + 0.1 * t
            y = rng.normal() + 0.2 * t + 0.3 * baseline * t + effect * (dose - baseline) + 0.5 * x
            rows.append((unit, t, dose, y, x, weight, (unit - 1) // 8 + 1))
    return pl.DataFrame(rows, schema=["g", "t", "d", "y", "x", "w", "cl"], orient="row")


@pytest.fixture(scope="module")
def didinter_continuous_csv_path(didinter_continuous_data, tmp_path_factory):
    """Write the continuous-baseline panel to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_continuous.csv"
    didinter_continuous_data.write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def didinter_continuous_gaps(didinter_continuous_data):
    """Continuous-baseline panel with some rows absent, and the same panel with those rows present but empty."""
    rng = np.random.default_rng(31)
    gone = pl.Series(rng.random(didinter_continuous_data.height) < 0.04) & (didinter_continuous_data["t"] > 1)
    absent = didinter_continuous_data.filter(~gone)
    empty = didinter_continuous_data.with_columns(
        pl.when(gone).then(None).otherwise(pl.col(name)).alias(name) for name in ["y", "d"]
    )
    return absent, empty


@pytest.fixture(scope="module")
def didinter_continuous_gaps_csv_paths(didinter_continuous_gaps, tmp_path_factory):
    """Write the panels with absent and with empty rows to CSV files and return their paths."""
    folder = tmp_path_factory.mktemp("didinter")
    paths = []
    for name, data in zip(["absent", "empty"], didinter_continuous_gaps, strict=True):
        path = folder / f"didinter_continuous_{name}.csv"
        data.write_csv(path, null_value="NA")
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def didinter_missing_outcome_data():
    """Clustered panel with treatment rises and falls whose gaps are missing outcomes instead of absent rows."""
    rng = np.random.default_rng(20261006)
    rows = []
    for unit in range(1, 301):
        baseline = int(rng.choice([0, 1, 2], p=[0.5, 0.25, 0.25]))
        switch = int(rng.integers(3, 7)) if rng.random() < 0.7 else None
        step = int(rng.choice([1, 2])) if baseline < 2 else -1
        slope = rng.normal(scale=0.2)
        weight = float(rng.uniform(0.5, 2.0))
        for t in range(1, 9):
            dose = baseline + step if switch is not None and t >= switch else baseline
            x = rng.normal() + 0.1 * t
            y = rng.normal() + slope * t + 1.2 * (dose - baseline) + 0.5 * x
            rows.append((unit, t, float(dose), y, x, weight, (unit - 1) // 10 + 1))
    df = pl.DataFrame(rows, schema=["g", "t", "d", "y", "x", "w", "cl"], orient="row")
    missing = pl.Series(rng.random(df.height) < 0.07)
    return df.with_columns(pl.when(missing).then(None).otherwise(pl.col("y")).alias("y"))


@pytest.fixture(scope="module")
def didinter_missing_outcome_csv_path(didinter_missing_outcome_data, tmp_path_factory):
    """Write the panel with missing outcomes to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_missing_outcome.csv"
    didinter_missing_outcome_data.write_csv(path, null_value="NA")
    return str(path)


@pytest.fixture(scope="module")
def didinter_het_data():
    """Clustered panel with a group covariate, group weights, and missing outcomes in the first period."""
    rng = np.random.default_rng(20261007)
    rows = []
    for unit in range(1, 301):
        switch = int(rng.choice([3, 4, 5, 6])) if unit > 80 else None
        x = float(rng.normal())
        weight = float(rng.uniform(0.5, 2.0))
        effect = rng.normal()
        for t in range(1, 9):
            dose = float(switch is not None and t >= switch)
            y = effect + 0.3 * t + (1.0 + 0.5 * x) * dose + rng.normal()
            rows.append((unit, t, dose, y, x, weight, (unit - 1) // 6 + 1))
    df = pl.DataFrame(rows, schema=["g", "t", "d", "y", "x", "w", "cl"], orient="row")
    missing = pl.Series(rng.random(df.height) < 0.05)
    return df.with_columns(pl.when(missing).then(None).otherwise(pl.col("y")).alias("y"))


@pytest.fixture(scope="module")
def didinter_het_csv_path(didinter_het_data, tmp_path_factory):
    """Write the heterogeneity panel to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_het.csv"
    didinter_het_data.write_csv(path, null_value="NA")
    return str(path)


@pytest.fixture(scope="module")
def didinter_baseline_shift_data():
    """Unbalanced panel with baseline treatment 0.1 and some bidirectional switchers, and the same panel at 0.125."""
    rng = np.random.default_rng(3)
    rows = []
    for unit in range(1, 201):
        switch = int(rng.integers(3, 8)) if rng.random() < 0.6 else None
        back = switch is not None and unit % 6 == 0 and switch + 2 <= 8
        effect = rng.normal()
        for t in range(1, 9):
            change = 0.0
            if switch is not None and t >= switch:
                change = -1.0 if back and t >= switch + 2 else 1.0
            x = rng.normal() + 0.1 * t
            y = effect + 0.1 * t + 0.6 * change + 0.5 * x + rng.normal()
            rows.append((unit, t, change, y, x, (unit - 1) // 10 + 1))
    df = pl.DataFrame(rows, schema=["g", "t", "change", "y", "x", "cl"], orient="row")
    df = df.filter(~(pl.Series(rng.random(df.height) < 0.08) & (df["t"] > 1)))
    return [df.with_columns((pl.col("change") + base).alias("d")).drop("change") for base in (0.1, 0.125)]


@pytest.fixture(scope="module")
def didinter_baseline_shift_csv_path(didinter_baseline_shift_data, tmp_path_factory):
    """Write the panel at baseline 0.125 to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "didinter_baseline_shift.csv"
    didinter_baseline_shift_data[1].write_csv(path)
    return str(path)


@pytest.fixture(scope="module")
def favara_missing_states():
    """Favara and Imbs data without the state in three rows of switching counties and in every row of 42 counties."""
    data = load_favara_imbs()
    switchers = data.filter(pl.col("inter_bra") > 0)["county"].unique().sort().to_list()[:3]
    rows = (
        ((pl.col("county") == switchers[0]) & (pl.col("year") == 1998))
        | ((pl.col("county") == switchers[1]) & (pl.col("year") == 2001))
        | ((pl.col("county") == switchers[2]) & (pl.col("year") == 1994))
    )
    counties = data["county"].unique().sort().to_list()[::20][:42]
    missing = rows | pl.col("county").is_in(counties)
    return data.with_columns(pl.when(missing).then(None).otherwise(pl.col("state_n")).alias("state_n"))


@pytest.fixture(scope="module")
def favara_missing_states_csv_path(favara_missing_states, tmp_path_factory):
    """Write the Favara and Imbs data with missing states to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "favara_missing_states.csv"
    favara_missing_states.write_csv(path, null_value="NA")
    return str(path)


@pytest.fixture(scope="module")
def favara_moved_county():
    """Favara and Imbs data in which county 1001 belongs to a second state from 2000 on."""
    moved = (pl.col("county") == 1001) & (pl.col("year") >= 2000)
    return load_favara_imbs().with_columns(pl.when(moved).then(2).otherwise(pl.col("state_n")).alias("state_n"))


@pytest.fixture(scope="module")
def favara_moved_county_csv_path(favara_moved_county, tmp_path_factory):
    """Write the Favara and Imbs data with the moved county to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("didinter") / "favara_moved_county.csv"
    favara_moved_county.write_csv(path)
    return str(path)


@pytest.fixture
def reference_draws():
    """Function that reads the cluster draws the reference saved in a folder as positions among the sorted clusters."""

    def read(folder, clusters):
        pool = np.sort(clusters.drop_nulls().unique().to_numpy())
        tables = [pl.read_csv(path) for path in sorted(folder.glob("rep_*.csv"))]
        return [
            np.repeat(np.searchsorted(pool, table["unit_id"].to_numpy()), table["count"].to_numpy()) for table in tables
        ]

    return read


@pytest.fixture(scope="module")
def mpdta_moderators():
    """Add the Great Lakes indicator, its complement, and an above-median employment indicator to mpdta."""
    gls_states = [17, 18, 26, 27, 36, 39, 42, 55]
    return (
        load_mpdta()
        .with_columns((pl.col("countyreal") // 1000).is_in(gls_states).alias("gls"))
        .with_columns(
            (~pl.col("gls")).alias("notgls"),
            (pl.col("lemp") > pl.col("lemp").median()).cast(pl.Int64).alias("ybin"),
        )
    )


@pytest.fixture(scope="module")
def mpdta_state_dummies():
    """Add an indicator for every state but the first to mpdta and return it with the formula of those indicators."""
    data = load_mpdta().with_columns((pl.col("countyreal") // 1000).alias("state"))
    states = sorted(data["state"].unique().to_list())[1:]
    data = data.with_columns([(pl.col("state") == s).cast(pl.Float64).alias(f"st{s}") for s in states])
    return data, "~" + " + ".join(f"st{s}" for s in states)


@pytest.fixture(scope="module")
def mpdta_dotted_names():
    """Add copies of the unit, period, control, and outcome columns of mpdta under dotted names."""
    return load_mpdta().with_columns(
        pl.col("countyreal").alias("county.id"),
        pl.col("year").alias("year.t"),
        pl.col("lpop").alias("log.pop"),
        pl.col("lemp").alias("l.emp"),
    )


@pytest.fixture(scope="module")
def mpdta_never_codes():
    """Code the never-treated counties of mpdta as missing, infinite, or 9999 instead of 0."""
    data = load_mpdta()
    never = pl.col("first.treat") == 0
    cohort = pl.col("first.treat").cast(pl.Float64)
    codes = {"null": None, "inf": float("inf"), "9999": 9999.0}
    return {
        label: data.with_columns(
            pl.when(never).then(pl.lit(code, dtype=pl.Float64)).otherwise(cohort).alias("first.treat")
        )
        for label, code in codes.items()
    }


@pytest.fixture(scope="module")
def mpdta_late_cohort():
    """Move the 2006 cohort of mpdta to 2008, after the last period."""
    data = load_mpdta()
    return data.with_columns(
        pl.when(pl.col("first.treat") == 2006).then(2008).otherwise(pl.col("first.treat")).alias("first.treat")
    )


@pytest.fixture(scope="module")
def mpdta_always_treated():
    """Relabel the last 30 never-treated counties of mpdta as a cohort treated in 2003, the first period."""
    data = load_mpdta()
    ids = data.filter(pl.col("first.treat") == 0)["countyreal"].unique().sort().tail(30)
    in_ids = pl.col("countyreal").is_in(ids.implode())
    return data.with_columns(pl.when(in_ids).then(2003).otherwise(pl.col("first.treat")).alias("first.treat"))


@pytest.fixture(scope="module")
def mpdta_missing_inputs():
    """Add county weights and the Great Lakes indicator to mpdta and return it with an expression for 158 rows."""
    data = load_mpdta().with_columns(
        (1 + (pl.col("countyreal") % 7) / 7).alias("w"),
        (pl.col("countyreal") // 1000).is_in([17, 18, 26, 27, 36, 39, 42, 55]).alias("gls"),
    )
    return data, (pl.col("countyreal") * 31 + pl.col("year")) % 17 == 0


@pytest.fixture(scope="module")
def mpdta_singletons():
    """Keep only the 2006 row of 15 never-treated and 10 early-treated counties of mpdta."""
    data = load_mpdta()
    never = data.filter(pl.col("first.treat") == 0)["countyreal"].unique().sort().head(15)
    early = data.filter(pl.col("first.treat") == 2004)["countyreal"].unique().sort().head(10)
    single = pl.col("countyreal").is_in(pl.concat([never, early]).implode())
    return data.filter(~single | (pl.col("year") == 2006))


@pytest.fixture
def rm_cases():
    """Return event studies with three pre-periods whose largest first difference sits in different places."""
    sd = np.array([0.004, 0.004, 0.004, 0.005, 0.006])
    sigma = (np.full((5, 5), 0.2) + 0.8 * np.eye(5)) * np.outer(sd, sd)
    return {
        "sigma": sigma,
        "A": np.array([-0.01, -0.005, -0.02, 0.05, 0.06]),
        "B": np.array([-0.03, -0.02, -0.005, 0.05, 0.06]),
        "C": np.array([-0.05, -0.005, -0.02, 0.05, 0.06]),
    }


@pytest.fixture(scope="module")
def mpdta_without_never_treated():
    """Keep the mpdta counties that are eventually treated."""
    return load_mpdta().filter(pl.col("first.treat") != 0)


@pytest.fixture(scope="module")
def mpdta_without_never_treated_csv_path(mpdta_without_never_treated, tmp_path_factory):
    """Write mpdta without its never-treated counties to a CSV file and return its path."""
    path = tmp_path_factory.mktemp("mpdta") / "mpdta_without_never_treated.csv"
    mpdta_without_never_treated.write_csv(path)
    return str(path)


@pytest.fixture
def mpdta_cohort_after_panel(request):
    """Move the 2007 cohort of mpdta to the year in request.param, after the last observed year."""
    return load_mpdta().with_columns(
        pl.when(pl.col("first.treat") == 2007).then(request.param).otherwise(pl.col("first.treat")).alias("first.treat")
    )


@pytest.fixture
def mpdta_cohort_after_panel_csv_path(mpdta_cohort_after_panel, tmp_path):
    """Write mpdta with the moved cohort to a CSV file and return its path."""
    path = tmp_path / "mpdta_cohort_after_panel.csv"
    mpdta_cohort_after_panel.write_csv(path)
    return str(path)


@pytest.fixture
def mp_first_period_cohort_data():
    """Multi-period panel in which a fifth of the units of cohort 2 are first treated in period 1."""
    data = gen_ddd_mult_periods(n=1000, random_state=7)["data"]
    early = (pl.col("group") == 2) & (pl.col("id") % 5 == 0)
    return data.with_columns(pl.when(early).then(1).otherwise(pl.col("group")).alias("group"))


@pytest.fixture
def mp_no_never_treated_data():
    """Panel over five periods whose units are all treated by period 4."""
    data = gen_ddd_scalable(n=3000, n_periods=5, n_cohorts=3, n_covariates=4, random_state=11)["data"]
    return data.filter(pl.col("group") != 0)


@pytest.fixture
def mp_no_never_treated_layout_data(request, mp_no_never_treated_data):
    """The panel without never-treated units with gaps from period 4 on, units seen only then, or one row per id."""
    data = mp_no_never_treated_data
    if request.param == "late_gap":
        return data.filter(~((pl.col("id") % 5 == 0) & (pl.col("time") == 5)))
    if request.param == "cohort_gap":
        return data.filter(~((pl.col("id") % 5 == 0) & (pl.col("time") == 4)))
    if request.param == "late_only":
        return data.filter(~((pl.col("id") % 7 == 0) & (pl.col("time") <= 3)))
    return data.with_columns(pl.int_range(pl.len()).alias("id"))


@pytest.fixture
def mp_ddd_weighted_data():
    """Multi-period panel with a sampling weight from U(0.2, 5) for each unit in w."""
    data = gen_ddd_mult_periods(n=1000, random_state=7)["data"]
    ids = data["id"].unique().sort()
    weights = pl.DataFrame({"id": ids, "w": np.random.default_rng(0).uniform(0.2, 5.0, len(ids))})
    return data.join(weights, on="id").sort("id", "time")


@pytest.fixture
def mp_ddd_weighted_unbalanced_data(mp_ddd_weighted_data):
    """Drop about 8 percent of the rows of the weighted multi-period panel to unbalance it."""
    keep = np.random.default_rng(7).random(mp_ddd_weighted_data.height) >= 0.08
    return mp_ddd_weighted_data.filter(pl.Series(keep))


@pytest.fixture
def mp_rcs_weighted_data(mp_ddd_weighted_data):
    """The weighted multi-period panel with every row taken as an observation of its own in rid."""
    return mp_ddd_weighted_data.with_columns(pl.int_range(pl.len()).alias("rid"))


@pytest.fixture
def mp_ddd_unbalanced_clustered_data(mp_ddd_clustered_data):
    """Drop about 8 percent of the rows of the clustered multi-period panel to unbalance it."""
    keep = np.random.default_rng(7).random(mp_ddd_clustered_data.height) >= 0.08
    return mp_ddd_clustered_data.filter(pl.Series(keep))


@pytest.fixture
def mp_rcs_clustered_data(mp_rcs_data):
    """Nest clusters of unequal size in the groups of the cross-section and shock the eligible observations of each."""
    data = mp_rcs_data.with_columns(
        (100 * pl.col("group") + (pl.col("id") % 300 + 1).sqrt().floor()).cast(pl.Int64).alias("cluster")
    )
    clusters = data["cluster"].unique().sort().to_list()
    shocks = dict(zip(clusters, np.random.default_rng(10).normal(0, 1, len(clusters))))
    shock = pl.col("cluster").replace_strict(shocks, return_dtype=pl.Float64)
    return data.with_columns(pl.col("y") + pl.col("partition") * shock)


@pytest.fixture
def mp_ddd_missing_value_data(request, mp_ddd_data):
    """Multi-period panel in which unit 3 misses its period-3 cohort or repeats period 1 without an outcome."""
    if request.param == "cohort":
        row = (pl.col("id") == 3) & (pl.col("time") == 3)
        return mp_ddd_data.with_columns(pl.when(row).then(None).otherwise(pl.col("group")).alias("group"))
    repeated = mp_ddd_data.filter((pl.col("id") == 3) & (pl.col("time") == 1))
    return pl.concat([mp_ddd_data, repeated.with_columns(pl.lit(None, pl.Float64).alias("y"))])
