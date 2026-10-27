import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import att_gt, compute_att_gt, gen_did_scalable, load_mpdta, mp, preprocess_did
from moderndid.core.preprocess import WEIGHTS_COLUMN
from moderndid.did.shards.container import RowShard
from moderndid.did.shards.local import LocalExecutor, row_shards_from_did_data
from moderndid.did.shards.partials import combine_partials, empty_partial, finalize_keyed
from moderndid.did.shards.row_driver import fit_att_gt_on_row_shards


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
    """mpdta with sampling weights that differ in every year of each county."""
    return mpdta_data.with_columns(
        (pl.col("lpop").exp() * (1 + (pl.col("countyreal") + pl.col("year")) % 5)).alias("w")
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
        return conn.sql("SELECT * FROM mpdta")
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
def mpdta_decimal_values(mpdta_pop_weighted):
    """Weighted mpdta with one missing outcome that stores the outcome and covariate and weights as decimals."""
    county = mpdta_pop_weighted.filter(pl.col("first.treat") == 2004)["countyreal"].min()
    row = (pl.col("countyreal") == county) & (pl.col("year") == 2005)
    holed = mpdta_pop_weighted.with_columns(pl.when(row).then(None).otherwise(pl.col("lemp")).alias("lemp"))
    return holed.with_columns(pl.col("lemp", "lpop", "pop").cast(pl.Decimal(18, 6)))


@pytest.fixture
def mpdta_large_ids(mpdta_data):
    """mpdta with 2**53 added to each county code."""
    return mpdta_data.with_columns((pl.col("countyreal") + 2**53).alias("countyreal"))


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


def _balanced_panel(cohorts, weights, covariate, n_periods, outcome):
    """Build a balanced panel from the cohort, weight, and covariate of each unit."""
    n_units = len(cohorts)
    ids = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)
    g = np.repeat(cohorts, n_periods)
    x = np.repeat(covariate, n_periods)
    return pl.DataFrame(
        {"id": ids, "t": t, "g": g, "w": np.repeat(weights, n_periods), "x": x, "y": outcome(ids, t, g, x)}
    )


@pytest.fixture
def overlap_flip_panel():
    """Panel whose weighted comparison units overlap the treated units while the unweighted ones nearly separate."""
    return _balanced_panel(
        np.r_[np.full(40, 3), np.zeros(104)],
        np.r_[np.ones(140), np.full(4, 40.0)],
        np.r_[np.linspace(0, 4, 40), np.linspace(-4, 0, 100), np.linspace(1, 2, 4)],
        3,
        lambda ids, t, g, x: x + t + np.sin(ids + 2 * t) + ((g == 3) & (t == 3)),
    )


@pytest.fixture
def thin_weighted_controls_panel():
    """Panel whose 10 never-treated units carry 0.004 of a treated unit's weight each."""
    return _balanced_panel(
        np.r_[np.full(50, 3), np.zeros(10)],
        np.r_[np.ones(50), np.full(10, 0.004)],
        np.zeros(60),
        3,
        lambda ids, t, g, x: t + np.cos(ids + t) + ((g == 3) & (t == 3)),
    )


@pytest.fixture
def zero_weight_controls_panel():
    """Panel whose never-treated units with a nonzero covariate all have zero weight."""
    return _balanced_panel(
        np.r_[np.full(30, 3), np.zeros(40)],
        np.r_[np.ones(50), np.zeros(20)],
        np.r_[np.linspace(-1, 1, 30), np.zeros(20), np.linspace(-1, 1, 20)],
        3,
        lambda ids, t, g, x: x + t + np.sin(ids + t),
    )


@pytest.fixture
def tiny_weight_overlap_panel():
    """Panel whose cohort 3 and never-treated units weigh about 1e-9 of a cohort 4 unit."""
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(1.35, 1.0, 40), rng.normal(-1.35, 1.0, 60)]
    weights = 1e-9 * np.exp(rng.normal(size=100))
    rng = np.random.default_rng(9)
    covariate = np.r_[x, rng.normal(size=40)]
    noise = rng.normal(size=4 * 140)
    return _balanced_panel(
        np.r_[np.full(40, 3), np.zeros(60), np.full(40, 4)],
        np.r_[weights, np.ones(40)],
        covariate,
        4,
        lambda ids, t, g, x: noise + x,
    )


@pytest.fixture
def tiny_late_cohort_panel():
    """Panel without never-treated units in which cohort 3 has 400 units and cohort 5 only 2."""
    return _balanced_panel(
        np.r_[np.full(400, 3), np.full(2, 5), np.full(30, 4)],
        np.ones(432),
        np.zeros(432),
        5,
        lambda ids, t, g, x: t + np.sin(1.3 * ids + t) + (t >= g),
    )


@pytest.fixture
def zero_weight_cohort_panel():
    """Panel whose cohort 3 has zero weight beside cohort 4 and never-treated units of weight 1."""
    return _balanced_panel(
        np.r_[np.full(20, 3), np.full(20, 4), np.zeros(20)],
        np.r_[np.zeros(20), np.ones(40)],
        np.tile(np.linspace(-1, 1, 20), 3),
        5,
        lambda ids, t, g, x: x + t + np.sin(ids + 2 * t) + ((g > 0) & (t >= g)),
    )


@pytest.fixture
def zero_weight_never_treated_panel():
    """Panel whose never-treated units have zero weight beside cohorts 3 and 4 of weight 1."""
    return _balanced_panel(
        np.r_[np.full(20, 3), np.full(20, 4), np.zeros(20)],
        np.r_[np.ones(40), np.zeros(20)],
        np.tile(np.linspace(-1, 1, 20), 3),
        5,
        lambda ids, t, g, x: x + t + np.sin(ids + 2 * t) + ((g > 0) & (t >= g)),
    )


@pytest.fixture
def zero_weight_treated_period_rcs():
    """Repeated cross sections whose cohort 3 rows in period 3 all have zero weight."""
    rng = np.random.default_rng(3)
    n_rows = 180
    t = np.repeat([1, 2, 3], 60)
    g = np.where(rng.random(n_rows) < 0.4, 3, 0)
    x = rng.normal(size=n_rows)
    treated_in_period_3 = (g == 3) & (t == 3)
    return pl.DataFrame(
        {
            "id": np.arange(n_rows),
            "t": t,
            "g": g,
            "x": x,
            "w": np.where(treated_in_period_3, 0.0, 1.0),
            "y": x + t + rng.normal(size=n_rows) + treated_in_period_3,
        }
    )


@pytest.fixture
def rcs_small_cohort_period():
    """Repeated cross sections with five covariates whose cohort 3 has 4 rows in period 2."""
    rng = np.random.default_rng(11)
    t = np.repeat([1, 1, 2, 2, 3, 3], [60, 140, 4, 140, 60, 140])
    g = np.repeat([3, 0, 3, 0, 3, 0], [60, 140, 4, 140, 60, 140])
    x = rng.normal(size=(t.size, 5))
    y = x @ np.arange(1, 6) + t + (g == 3) * (t == 3) + rng.normal(size=t.size)
    return pl.DataFrame({"id": np.arange(t.size), "t": t, "g": g, "y": y, **{f"x{j}": x[:, j] for j in range(5)}})


@pytest.fixture
def no_post_treatment_result():
    """att_gt result whose only post-treatment cell fails the overlap screen."""
    covariate_by_period = {
        1: np.sin(np.arange(100.0)),
        2: np.r_[np.full(40, 4.0), np.full(60, -4.0)] + 0.3 * np.cos(np.arange(100.0)),
    }
    covariate_by_period[3] = covariate_by_period[2]
    cohorts = np.r_[np.full(40, 3), np.zeros(60)]
    rows = [
        {
            "id": unit,
            "t": t,
            "g": cohorts[unit],
            "x": covariate_by_period[t][unit],
            "y": 0.5 * covariate_by_period[t][unit] + np.cos(unit + t) + float(cohorts[unit] == 3 and t == 3),
        }
        for unit in range(100)
        for t in (1, 2, 3)
    ]
    with pytest.warns(UserWarning, match="^Overlap condition violated for 3 in time period 3$"):
        return att_gt(pl.DataFrame(rows), yname="y", tname="t", idname="id", gname="g", xformla="~x")


@pytest.fixture
def mpdta_separated_2004(mpdta_data):
    """mpdta with a covariate that separates the 2004 cohort from every other county."""
    return mpdta_data.with_columns((pl.col("lpop") + 100 * (pl.col("first.treat") == 2004)).alias("size"))


@pytest.fixture
def mpdta_with_states(mpdta_data):
    """mpdta with the state of each county as a cluster variable."""
    return mpdta_data.with_columns((pl.col("countyreal") // 1000).alias("state"))


@pytest.fixture
def mpdta_varying_weights_and_states(mpdta_varying_weights):
    """mpdta with sampling weights that vary over time and the state of each county as a cluster variable."""
    return mpdta_varying_weights.with_columns((pl.col("countyreal") // 1000).alias("state"))


@pytest.fixture
def mpdta_collinear_covariates(mpdta_data):
    """mpdta with a covariate that is exactly twice lpop."""
    return mpdta_data.with_columns((2 * pl.col("lpop")).alias("lpop2"))


@pytest.fixture
def scalable_panel():
    """Generated panel of 400 units over five periods with four covariates and a standardized outcome."""
    data = gen_did_scalable(400, n_periods=5, n_cohorts=3, n_covariates=4, random_state=42)["data"]
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std())


@pytest.fixture
def large_scalable_panel():
    """Generated panel of 2500 units over three periods with 50 clusters and a standardized outcome."""
    data = gen_did_scalable(2500, n_periods=3, n_cohorts=2, n_covariates=4, random_state=42)["data"]
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std())


@pytest.fixture
def mp_result():
    df = load_mpdta()

    result = att_gt(
        data=df,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        xformla="~ 1",
        est_method="reg",
    )

    return result


@pytest.fixture
def mpdta_mp_result():
    mpdta = load_mpdta()
    mpdta = mpdta.with_columns((pl.col("countyreal") % 10).alias("cluster"))

    data = preprocess_did(
        mpdta,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        anticipation=0,
        clustervars=["cluster"],
    )

    att_result = compute_att_gt(data)

    groups = np.array([r.group for r in att_result.attgt_list])
    times = np.array([r.year for r in att_result.attgt_list])
    att_gt = np.array([r.att for r in att_result.attgt_list])

    inf_func = att_result.influence_functions.toarray()
    n_units = data.config.id_count

    vcov = np.cov(inf_func.T)
    se_gt = np.sqrt(np.diag(vcov) / n_units)

    gname = data.config.gname
    G = data.time_invariant_data[gname].to_numpy()
    weights_ind = np.ones(n_units)

    mp_result = mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=n_units,
        alpha=0.05,
        estimation_params={
            "bootstrap": False,
            "uniform_bands": False,
            "control_group": "nevertreated",
            "anticipation_periods": 0,
            "estimation_method": "dr",
            "data": data,
            "panel": True,
        },
        G=G,
        weights_ind=weights_ind,
    )

    return mp_result


@pytest.fixture
def synthetic_mp_result():
    groups = []
    times = []
    att_gt = []

    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006, 2007]:
            groups.append(g)
            times.append(t)
            if g <= t:
                att_gt.append(0.05 + 0.01 * (t - g))
            else:
                att_gt.append(0.0)

    groups = np.array(groups)
    times = np.array(times)
    att_gt = np.array(att_gt)

    n_units = 500
    n_gt = len(groups)
    inf_func = np.random.normal(0, 0.01, size=(n_units, n_gt))

    vcov = np.cov(inf_func.T)
    se_gt = np.sqrt(np.diag(vcov) / n_units)

    G = np.tile([2004, 2006, 2007, 0], n_units // 4)
    weights_ind = np.ones(n_units)

    return mp(
        groups=groups,
        times=times,
        att_gt=att_gt,
        vcov_analytical=vcov,
        se_gt=se_gt,
        critical_value=1.96,
        influence_func=inf_func,
        n_units=n_units,
        alpha=0.05,
        estimation_params={
            "bootstrap": False,
            "uniform_bands": False,
            "control_group": "nevertreated",
            "anticipation_periods": 0,
            "estimation_method": "dr",
        },
        G=G,
        weights_ind=weights_ind,
    )


@pytest.fixture
def panel_data(mpdta_data):
    return preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        est_method="dr",
    )


@pytest.fixture
def _preprocessed_panel(mpdta_data):
    return preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~lpop",
        panel=True,
        control_group="nevertreated",
        base_period="varying",
    )


@pytest.fixture
def panel_df():
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": [0, 0, 3, 3, 3, 3],
        }
    )


@pytest.fixture
def rcs_df():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "time": [1, 1, 1, 2, 2, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": [0, 3, 3, 0, 3, 3],
        }
    )


class WholeClusterExecutor(LocalExecutor):
    """Executor that finalizes the keyed sums of each shard as a Spark task that holds whole clusters does."""

    def run(self, kernel, state):
        """Apply a kernel to every shard and finalize each shard's keyed sums before combining them."""
        self.n_passes += 1
        partial = empty_partial()
        for shard in self.shards:
            partial = combine_partials(partial, finalize_keyed(kernel(shard, state)))
        return finalize_keyed(partial)


def _whole_cluster_shards(shard, n_shards):
    """Split a row shard into shards that each hold every unit of their clusters."""
    _, codes = np.unique(shard.cluster_keys, return_inverse=True)
    owner = codes.ravel() % n_shards
    shards = []
    for part in range(n_shards):
        members = np.flatnonzero(owner == part)
        rows, units = members, None
        if shard.units is not None:
            rows = np.flatnonzero(np.isin(shard.units, members))
            units = np.searchsorted(members, shard.units[rows])
        shards.append(
            RowShard(
                period=shard.period[rows],
                cohort=shard.cohort[rows],
                y=shard.y[rows],
                weights=None if shard.weights is None else shard.weights[rows],
                covariates=None if shard.covariates is None else shard.covariates[rows],
                unit_keys=None if shard.unit_keys is None else shard.unit_keys[members],
                units=units,
                cluster_keys=shard.cluster_keys[members],
            )
        )
    return shards


@pytest.fixture
def att_gt_on_row_shards():
    """Function that runs att_gt on row shards of repeated cross-sections or an unbalanced panel in memory."""

    def fit(data, n_shards, shard_seed=0, random_state=None, keyed=True, whole_clusters=False, **spec):
        settings = {
            "alpha": 0.05,
            "cband": True,
            "boot": False,
            "biters": 1000,
            "clustervars": None,
            "est_method": "dr",
            "panel": True,
            "allow_unbalanced_panel": False,
            "control_group": "nevertreated",
            "anticipation": 0,
            "base_period": "varying",
            **spec,
        }
        did_data = preprocess_did(data, **settings)
        idname = settings.get("idname")
        clustervars = settings["clustervars"]
        cluster = next((name for name in clustervars or [] if name != idname), None)
        cluster_counts = None
        if cluster is not None:
            units = did_data.time_invariant_data
            per_cohort = units.group_by(settings["gname"]).agg(pl.col(cluster).n_unique())
            cluster_counts = (units[cluster].n_unique(), dict(per_cohort.iter_rows()))
        estimation_params = {
            "yname": settings["yname"],
            "idname": idname,
            "control_group": settings["control_group"],
            "anticipation_periods": settings["anticipation"],
            "estimation_method": settings["est_method"],
            "bootstrap": settings["boot"],
            "uniform_bands": settings["cband"],
            "base_period": settings["base_period"],
            "panel": settings["panel"],
            "clustervars": None if clustervars is None else [name for name in clustervars if name != idname],
            "cluster": None,
            "biters": settings["biters"],
            "random_state": random_state,
            "n_units": did_data.config.id_count,
            "n_obs": len(did_data.data),
        }
        if whole_clusters:
            whole = row_shards_from_did_data(did_data, 1, random_state=shard_seed, keyed=keyed, cluster=cluster)[0]
            executor = WholeClusterExecutor(_whole_cluster_shards(whole, n_shards))
        else:
            shards = row_shards_from_did_data(did_data, n_shards, random_state=shard_seed, keyed=keyed, cluster=cluster)
            executor = LocalExecutor(shards)
        return fit_att_gt_on_row_shards(
            executor,
            did_data.config,
            did_data.cohort_counts,
            random_state,
            estimation_params,
            cluster_counts=cluster_counts,
        )

    return fit


@pytest.fixture
def record_warnings():
    """Function that returns a call's result with the category and message of every warning it raised."""

    def call(function, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = function(*args, **kwargs)
        return result, [(warning.category, str(warning.message)) for warning in caught]

    return call


@pytest.fixture
def assert_same_fit():
    """Function that asserts that two att_gt results hold the same cells and estimates up to a tolerance."""

    def check(result, expected, tolerance):
        np.testing.assert_array_equal(result.groups, expected.groups)
        np.testing.assert_array_equal(result.times, expected.times)
        assert (result.groups.dtype, result.times.dtype) == (expected.groups.dtype, expected.times.dtype)
        assert result.n_units == expected.n_units
        np.testing.assert_allclose(result.att_gt, expected.att_gt, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.se_gt, expected.se_gt, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.vcov_analytical, expected.vcov_analytical, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(result.critical_value, expected.critical_value, rtol=tolerance)
        wald = [np.nan if fit.wald_stat is None else fit.wald_stat for fit in (result, expected)]
        np.testing.assert_allclose(wald[0], wald[1], rtol=tolerance)

    return check


@pytest.fixture
def splitmix_row_keys():
    """Function that keys rows by the splitmix64 chain of their 64-bit words and copy indices in Python integers."""

    def mix(z):
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 % 2**64
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB % 2**64
        return z ^ (z >> 31)

    def keys(period, floats, copy=None):
        words = [np.asarray(period, dtype=np.int64).astype(np.uint64).tolist()]
        words += [(np.asarray(values, dtype=np.float64) + 0.0).view(np.uint64).tolist() for values in floats]
        seen = {}
        chains = []
        for position, row in enumerate(zip(*words)):
            index = seen.get(row, 0) if copy is None else int(copy[position])
            seen[row] = seen.get(row, 0) + 1
            chain = 0x9E3779B97F4A7C15
            for word in (*row, index):
                chain = mix(chain ^ word)
            chains.append(chain)
        return np.array(chains, dtype=np.uint64)

    return keys


@pytest.fixture
def unit_table():
    """Function that lays out a preprocessed unbalanced panel as the Arrow unit table that a Spark fit reads."""

    def build(did_data):
        config = did_data.config
        covariates = np.asarray(did_data.covariates_matrix)[:, 1:]
        rows = did_data.data.with_columns(
            pl.Series("period", np.searchsorted(config.time_periods, did_data.data[config.tname].to_numpy())),
            pl.Series("x", covariates.tolist(), dtype=pl.List(pl.Float64)),
        ).sort(config.idname, "period")
        units = rows.group_by(config.idname, maintain_order=True).agg(
            pl.col(config.gname).first().cast(pl.Float64).alias("cohort"),
            pl.col("period"),
            pl.col(config.yname).cast(pl.Float64).alias("y"),
            pl.col(WEIGHTS_COLUMN).alias("w"),
            pl.col("x").explode(),
        )
        return units.rename({config.idname: "unit"}).to_arrow()

    return build


@pytest.fixture
def mpdta_thinned(mpdta_varying_weights_and_states):
    """mpdta with time-varying weights and states that keeps a random 80 percent of its rows."""
    data = mpdta_varying_weights_and_states
    rows = np.sort(np.random.default_rng(0).choice(data.height, size=4 * data.height // 5, replace=False))
    return data[rows]


@pytest.fixture(params=["repeated_cross_sections", "unbalanced_panel"])
def mpdta_layout(request, mpdta_varying_weights_and_states, mpdta_thinned):
    """mpdta with time-varying weights and states as repeated cross-sections or as a thinned unbalanced panel."""
    if request.param == "repeated_cross_sections":
        return mpdta_varying_weights_and_states, {"panel": False}
    return mpdta_thinned, {"allow_unbalanced_panel": True}


@pytest.fixture
def mpdta_layout_with_cohort_gaps(mpdta_layout):
    """mpdta of either layout in which cohorts 2004 and 2006 and the never-treated counties miss some periods."""
    data, layout = mpdta_layout
    gaps = (
        ((pl.col("first.treat") == 2006) & (pl.col("year") == 2005))
        | ((pl.col("first.treat") == 0) & (pl.col("year") == 2004))
        | ((pl.col("first.treat") == 2004) & pl.col("year").is_in([2003, 2005]))
    )
    return data.filter(~gaps), layout


@pytest.fixture
def mpdta_single_row_units(mpdta_thinned):
    """Thinned mpdta without never-treated rows of 2007 in which some counties keep one row of 2005 or of 2007."""
    counties = np.random.default_rng(3).permutation(mpdta_thinned["countyreal"].unique().sort().to_numpy())
    only_2007, only_2005 = counties[:30].tolist(), counties[30:70].tolist()
    return mpdta_thinned.filter(
        ~((pl.col("first.treat") == 0) & (pl.col("year") == 2007))
        & (~pl.col("countyreal").is_in(only_2007) | (pl.col("year") == 2007))
        & (~pl.col("countyreal").is_in(only_2005) | (pl.col("year") == 2005))
    )


@pytest.fixture
def mpdta_layout_with_one_positive_comparison_weight(mpdta_layout):
    """mpdta of either layout whose never-treated rows of 2003 all weigh zero but one."""
    data, layout = mpdta_layout
    never_2003 = (pl.col("first.treat") == 0) & (pl.col("year") == 2003)
    county = data.filter(never_2003)["countyreal"].min()
    zero = never_2003 & (pl.col("countyreal") != county)
    return data.with_columns(pl.when(zero).then(0.0).otherwise(pl.col("w")).alias("w")), layout


@pytest.fixture
def mpdta_layout_with_extreme_weight_ratio(mpdta_layout):
    """mpdta of either layout with zero weights among never-treated rows and positive weights 1.5e6 apart in 2004."""
    data, layout = mpdta_layout
    never = pl.col("first.treat") == 0
    zero = never & pl.col("year").is_in([2004, 2006]) & (pl.col("countyreal") % 7 == 0)
    positive_2004 = never & (pl.col("year") == 2004) & ~zero
    smallest = data.filter(positive_2004)["w"].min()
    heaviest = data.filter(positive_2004).sort("w")["countyreal"][-1]
    heavy = positive_2004 & (pl.col("countyreal") == heaviest)
    weights = pl.when(heavy).then(1.5e6 * smallest).when(zero).then(0.0).otherwise(pl.col("w"))
    return data.with_columns(weights.alias("w")), layout


@pytest.fixture
def mpdta_layout_with_zero_weight_periods(mpdta_layout):
    """mpdta of either layout whose cohort 2006 weighs zero in 2005 and never-treated counties weigh zero in 2006."""
    data, layout = mpdta_layout
    zero = ((pl.col("first.treat") == 2006) & (pl.col("year") == 2005)) | (
        (pl.col("first.treat") == 0) & (pl.col("year") == 2006)
    )
    return data.with_columns(pl.when(zero).then(0.0).otherwise(pl.col("w")).alias("w")), layout


@pytest.fixture
def mpdta_layout_with_few_clusters(mpdta_layout):
    """mpdta of either layout whose cohort 2004 falls in three clusters while the other counties cluster by state."""
    data, layout = mpdta_layout
    few = pl.when(pl.col("first.treat") == 2004).then(pl.col("countyreal") % 3).otherwise(pl.col("state"))
    return data.with_columns(few.alias("cluster")), layout


@pytest.fixture
def mpdta_row_clusters(mpdta_varying_weights_and_states):
    """mpdta with time-varying weights whose cluster changes from year to year within each county."""
    return mpdta_varying_weights_and_states.with_columns(
        ((7 * pl.col("countyreal") + pl.col("year")) % 23).alias("cluster")
    )


@pytest.fixture
def mpdta_with_duplicate_rows(mpdta_varying_weights_and_states):
    """mpdta with time-varying weights and repeated rows including five pairs whose zero outcomes differ in sign."""
    data = mpdta_varying_weights_and_states
    rows = np.random.default_rng(1).choice(data.height, size=50, replace=False)
    zeros = data[rows[:5]].with_columns(pl.lit(0.0).alias("lemp"))
    return pl.concat([data, data[rows], data[rows[:10]], zeros, zeros.with_columns(pl.lit(-0.0).alias("lemp"))])


@pytest.fixture
def mpdta_layout_with_collinear_covariates(mpdta_layout):
    """mpdta of either layout with twice lpop and with 1e7 times lpop plus 3e9 as further covariates."""
    data, layout = mpdta_layout
    return data.with_columns((2 * pl.col("lpop")).alias("lpop2"), (1e7 * pl.col("lpop") + 3e9).alias("big")), layout


@pytest.fixture
def mpdta_rounded_collinear_covariate(mpdta_data):
    """mpdta with a covariate that equals 1e7 times lpop plus 3e9 up to rounding."""
    return mpdta_data.with_columns((1e7 * pl.col("lpop") + 3e9).alias("big"))


@pytest.fixture(params=["repeated_cross_sections", "unbalanced_panel"])
def scalable_layout(request):
    """Generated data with four covariates and a standardized outcome as repeated cross-sections or a thinned panel."""
    if request.param == "repeated_cross_sections":
        data = gen_did_scalable(300, n_periods=5, n_cohorts=3, n_covariates=4, panel=False, random_state=42)["data"]
        layout = {"panel": False}
    else:
        data = gen_did_scalable(400, n_periods=5, n_cohorts=3, n_covariates=4, random_state=42)["data"]
        data = data.filter(pl.Series(np.random.default_rng(2).random(data.height) < 0.7))
        layout = {"allow_unbalanced_panel": True}
    return data.with_columns((pl.col("y") - pl.col("y").mean()) / pl.col("y").std()), layout


@pytest.fixture(params=["repeated_cross_sections", "unbalanced_panel"])
def far_comparison_unit(request):
    """Data of either layout whose one far never-treated unit overflows the logit of every cell."""
    rng = np.random.default_rng(3)
    ids = np.repeat(np.arange(121), 3)
    t = np.tile([1, 2, 3], 121)
    g = np.where(ids < 60, 3, 0)
    x = np.where(ids == 120, -1000.0, rng.normal(0.8 * (g == 3), 1.0))
    y = rng.normal(size=ids.size) + t + 0.5 * x * (ids < 120) + ((g == 3) & (t == 3))
    data = pl.DataFrame({"id": ids, "t": t, "g": g, "x": x, "y": y})
    if request.param == "repeated_cross_sections":
        return data, {"panel": False}
    keep = (np.random.default_rng(4).random(data.height) < 0.8) | (ids == 120)
    return data.filter(pl.Series(keep)), {"idname": "id", "allow_unbalanced_panel": True}


@pytest.fixture
def far_comparison_cell():
    """Treatment flags and covariate of one cell whose far comparison row overflows each iteration of its logit."""
    rng = np.random.default_rng(0)
    treated = np.r_[np.ones(60, dtype=bool), np.zeros(61, dtype=bool)]
    return treated, np.r_[rng.normal(0.8, 1.0, 60), rng.normal(0.0, 1.0, 60), -1000.0]


@pytest.fixture
def degenerate_comparison_rows(request):
    """Rows of cohort 3 and never-treated units whose comparison rows of period 2 the parameter makes degenerate."""
    kind, layout = request.param
    rng = np.random.default_rng(3)
    rows = []
    for t in (1, 2, 3, 4):
        for g in (3, 0):
            degenerate = g == 0 and t == 2
            for k in range(2 if degenerate and kind == "two_rows" else 10):
                x = float(2 * k - 1) if degenerate and kind == "two_rows" else float(rng.integers(-3, 4))
                y = rng.normal() + t + 0.5 * x + 1.5 * (g == 3 and t >= 3)
                if degenerate and kind == "constant":
                    y = 1.0
                if degenerate and kind == "exact":
                    y = 2.0 + 3.0 * x
                rows.append((k + 100 * (g == 3), t, g, x, y))
    data = pl.DataFrame(rows, schema=["unit", "t", "g", "x1", "y"], orient="row")
    if layout == "repeated_cross_sections":
        return data.with_row_index("id"), {"panel": False}
    unbalanced = data.rename({"unit": "id"}).filter(~((pl.col("t") == 4) & (pl.col("id") % 10 == 9)))
    return unbalanced, {"allow_unbalanced_panel": True}


@pytest.fixture
def degenerate_comparison_panel(request):
    """Balanced panel of cohort 3 and never-treated units whose comparison changes the parameter makes degenerate."""
    kind = request.param
    rng = np.random.default_rng(5)
    rows = []
    for unit in range(32 if kind == "two_comparison_units" else 42):
        cohort = 3 if unit < 30 else 0
        level = float(rng.integers(0, 10))
        x = float(rng.integers(-3, 4))
        weight = 1.0 + unit % 3
        if kind == "two_comparison_units" and cohort == 0:
            x = float(unit - 30)
        for t in (1, 2, 3, 4):
            if cohort == 0 and kind == "constant_change":
                y = level + 2.0 * t
            elif cohort == 0 and kind == "exact_change":
                y = level + t * (1.0 + 2.0 * x)
            else:
                y = level + t + 0.5 * x * t + rng.normal() + 1.5 * (cohort == 3 and t >= 3)
            rows.append((unit, t, cohort, x, weight, y))
    return pl.DataFrame(rows, schema=["id", "t", "g", "x1", "w", "y"], orient="row")
