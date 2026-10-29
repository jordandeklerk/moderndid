import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import att_gt, compute_att_gt, gen_did_scalable, load_mpdta, mp, preprocess_did


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


@pytest.fixture
def mean_change_cells():
    """Function that returns each cell's weighted difference in mean outcome changes and its unit influence function."""

    def estimate(data, cells, weightsname=None, yname="lemp", tname="year", idname="countyreal", gname="first.treat"):
        units = data.unique(idname).sort(idname)
        cohorts = units[gname].to_numpy()
        att = np.zeros(len(cells))
        influence = np.zeros((units.height, len(cells)))
        for k, (cohort, later, earlier, comparison) in enumerate(cells):
            rows = {period: data.filter(pl.col(tname) == period).sort(idname) for period in (later, earlier)}
            change = rows[later][yname].to_numpy() - rows[earlier][yname].to_numpy()
            weight = rows[min(later, earlier)][weightsname].to_numpy() if weightsname else np.ones(units.height)
            for side, sign in ((cohorts == cohort, 1.0), (np.isin(cohorts, comparison), -1.0)):
                mean = np.sum(weight[side] * change[side]) / np.sum(weight[side])
                att[k] += sign * mean
                influence[side, k] = sign * units.height * weight[side] * (change[side] - mean) / np.sum(weight[side])
        return att, influence

    return estimate


@pytest.fixture
def period_mean_cells():
    """Function that returns each cell's difference in four period means and its influence function by row or unit."""

    def estimate(data, cells, idname=None, yname="lemp", tname="year", gname="first.treat"):
        rows = data.sort(idname) if idname else data
        n_obs = rows[idname].n_unique() if idname else rows.height
        y, periods, cohorts = (rows[column].to_numpy() for column in (yname, tname, gname))
        att = np.zeros(len(cells))
        influence = np.zeros((rows.height, len(cells)))
        for k, (cohort, later, earlier, comparison) in enumerate(cells):
            treated = cohorts == cohort
            compared = np.isin(cohorts, comparison)
            for side, sign in (
                (treated & (periods == later), 1.0),
                (treated & (periods == earlier), -1.0),
                (compared & (periods == later), -1.0),
                (compared & (periods == earlier), 1.0),
            ):
                mean = y[side].mean()
                att[k] += sign * mean
                influence[side, k] = sign * n_obs * (y[side] - mean) / side.sum()
        if idname is None:
            return att, influence
        unit = np.unique(rows[idname].to_numpy(), return_inverse=True)[1]
        return att, np.stack([np.bincount(unit, weights=column) for column in influence.T], axis=1)

    return estimate


@pytest.fixture
def collinear_pretrend_panel():
    """Panel whose treated cohorts follow exact trends so that two pre-treatment cells share an influence function."""
    rng = np.random.default_rng(3)
    cohorts = np.r_[np.zeros(30), np.full(20, 3), np.full(20, 4)]
    level = rng.normal(size=cohorts.size)
    noise = rng.normal(size=(4, cohorts.size))
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "id": np.arange(cohorts.size),
                    "t": t,
                    "g": cohorts,
                    "y": level + np.where(cohorts == 0, noise[t - 1], 0.5 * t) + ((cohorts > 0) & (t >= cohorts)),
                }
            )
            for t in (1, 2, 3, 4)
        ]
    )


@pytest.fixture
def mpdta_cells():
    """Function that lists each mpdta cell as its cohort and period and base period and comparison cohorts."""

    def cells(base_period="varying", control_group="nevertreated", anticipation=0):
        cohorts = [group for group in (2004, 2006, 2007) if group > 2003 + anticipation]
        listed = []
        for group in cohorts:
            for period in range(2003 if base_period == "universal" else 2004, 2008):
                base = period - 1 if base_period == "varying" and period < group else group - 1 - anticipation
                later = max(period, base) + anticipation
                not_yet = [other for other in cohorts if other > later and other != group]
                listed.append((group, period, base, [0, *not_yet] if control_group == "notyettreated" else [0]))
        return listed

    return cells


@pytest.fixture
def four_means():
    """Function that computes a weighted difference in mean changes and the share of each row in its derivative."""

    def compute(y, treated, control, later, earlier, weights):
        att = 0.0
        share = np.zeros(len(y))
        parts = ((treated & later, 1), (treated & earlier, -1), (control & later, -1), (control & earlier, 1))
        for part, sign in parts:
            mean = np.sum(weights[part] * y[part]) / np.sum(weights[part])
            att += sign * mean
            share += sign * part * weights * (y - mean) / np.sum(weights[part])
        return att, share

    return compute


@pytest.fixture
def cells_from_weighted_means(four_means):
    """Function that computes the ATT and influence function of mpdta cells from weighted means of lemp."""

    def compute(rows, keys, cells, weights="pop"):
        lemp = rows["lemp"].to_numpy()
        year = rows["year"].to_numpy()
        cohort = rows["first.treat"].to_numpy()
        row_of = {key: position for position, key in enumerate(keys.iter_rows())}
        positions = np.array([row_of[key] for key in rows.select(keys.columns).iter_rows()])
        atts = []
        columns = []
        for group, period, base, controls in cells:
            att, share = four_means(
                lemp, cohort == group, np.isin(cohort, controls), year == period, year == base, rows[weights].to_numpy()
            )
            atts.append(att)
            columns.append(len(row_of) * np.bincount(positions, weights=share, minlength=len(row_of)))
        return np.array(atts), np.column_stack(columns)

    return compute


@pytest.fixture
def cells_from_estimator():
    """Function that runs a 2x2 estimator on the units of each mpdta cell and spreads its influence function."""

    def compute(rows, keys, cells, estimator, panel=True, weights="pop"):
        row_of = {key: position for position, key in enumerate(keys.iter_rows())}
        atts = []
        columns = []
        for group, period, base, controls in cells:
            if period == base:
                atts.append(0.0)
                columns.append(np.zeros(len(row_of)))
                continue
            cell = rows.filter(pl.col("first.treat").is_in([group, *controls]) & pl.col("year").is_in([period, base]))
            if panel:
                at_period = cell.filter(pl.col("year") == period).sort("countyreal")
                at_base = cell.filter(pl.col("year") == base).sort("countyreal")
                units = at_period if period < base else at_base
                outcomes = {"y1": at_period["lemp"].to_numpy(), "y0": at_base["lemp"].to_numpy()}
            else:
                units = cell
                outcomes = {"y": cell["lemp"].to_numpy(), "post": (cell["year"] == period).cast(pl.Int64).to_numpy()}
            fit = estimator(
                **outcomes,
                d=(units["first.treat"] == group).cast(pl.Float64).to_numpy(),
                covariates=np.column_stack([np.ones(units.height), units["lpop"].to_numpy()]),
                i_weights=None if weights is None else units[weights].to_numpy(),
                influence_func=True,
            )
            positions = [row_of[key] for key in units.select(keys.columns).iter_rows()]
            column = np.zeros(len(row_of))
            column[positions] = len(row_of) / len(positions) * fit.att_inf_func
            atts.append(fit.att)
            columns.append(column)
        return np.array(atts), np.column_stack(columns)

    return compute


@pytest.fixture
def mpdta_atts_with_scaled_unit_weights(mpdta_pop_weighted):
    """Function that refits the mpdta cells after scaling the pop weight of each county by its own factor."""

    def refit(units, factors, **spec):
        scaled = mpdta_pop_weighted.with_columns(pl.col("pop") * pl.col("countyreal").replace_strict(units, factors))
        return np.array([cell.att for cell in compute_att_gt(preprocess_did(scaled, **spec)).attgt_list])

    return refit


@pytest.fixture
def rcs_with_empty_cell_periods():
    """Repeated cross sections whose cohort 3 misses period 2 and whose never-treated rows miss periods 1 and 4."""
    rng = np.random.default_rng(13)
    present = {3: [1, 3, 4, 5], 5: [1, 2, 3, 4, 5], 0: [2, 3, 5]}
    cohort, period = np.array([(g, t) for g, periods in present.items() for t in periods for _ in range(30)]).T
    return pl.DataFrame(
        {"t": period, "g": cohort, "y": period + (cohort > 0) * (period >= cohort) + rng.normal(size=len(period))}
    )


@pytest.fixture
def zero_weight_outlier_panel():
    """Panel whose one comparison unit beyond the covariate range of the treated units has zero weight."""
    return _balanced_panel(
        np.r_[np.full(40, 3), np.zeros(61)],
        np.r_[np.ones(100), 0.0],
        np.r_[np.linspace(0, 2, 40), np.linspace(-1, 1, 60), 8.0],
        3,
        lambda ids, t, g, x: x + t + np.sin(ids + 2 * t) + ((g == 3) & (t == 3)),
    )


@pytest.fixture
def weighted_wide_panel():
    """Generated panel of 600 units over three periods with twenty covariates and unequal unit weights."""
    data = gen_did_scalable(600, n_periods=3, n_cohorts=2, n_covariates=20, random_state=42)["data"]
    weights = np.random.default_rng(42).uniform(0.2, 3.0, 600)
    return data.with_columns(pl.Series("w", weights[data["id"].to_numpy() - 1]))


@pytest.fixture
def logit_scores():
    """Function that fits a weighted logit by Newton's method and returns its fitted scores."""

    def fit(design, treated, weights):
        coefficients = np.zeros(design.shape[1])
        for _ in range(100):
            scores = 1 / (1 + np.exp(-design @ coefficients))
            gradient = design.T @ (weights * (treated - scores))
            hessian = design.T @ ((weights * scores * (1 - scores))[:, None] * design)
            coefficients = coefficients + np.linalg.solve(hessian, gradient)
        return 1 / (1 + np.exp(-design @ coefficients))

    return fit


@pytest.fixture
def wide_trimming_panel():
    """Panel with twenty covariates whose dummy clusters give comparison units 6 and 13 scores 840/841 and 120/121."""
    rng = np.random.default_rng(7)
    cohorts = np.r_[np.full(6, 3), 0, np.full(6, 3), 0, np.full(120, 3), np.zeros(240)]
    weights = np.r_[np.full(6, 140.0), 1.0, np.full(6, 20.0), 1.0, rng.uniform(0.5, 2.0, 360)]
    covariates = np.zeros((374, 20))
    covariates[:7, 0] = 1.0
    covariates[7:14, 1] = 1.0
    covariates[14:, 2:] = rng.normal(size=(360, 18))
    covariates[14:, 2] += 0.8 * (cohorts[14:] == 3)
    drift = np.zeros(374)
    drift[[6, 13]] = [6.0, -4.0]
    level = rng.normal(size=374)
    t = np.tile([1, 2, 3], 374)
    unit = np.repeat(np.arange(374), 3)
    treated_after = (cohorts[unit] == 3) & (t == 3)
    y = level[unit] + t * covariates[unit, 2] + (t - 1) * drift[unit] + rng.normal(size=t.size) + 2.0 * treated_after
    return pl.DataFrame(
        {
            "id": unit,
            "t": t,
            "g": cohorts[unit],
            "w": weights[unit],
            "y": y,
            **{f"x{j}": covariates[unit, j] for j in range(20)},
        }
    )


def _with_unit_draws(data, n_units, n_periods):
    """Add a uniform weight w and a standard normal draw noise for each unit of generated data sorted by unit."""
    rng = np.random.default_rng(1)
    weights = rng.uniform(0.5, 1.5, n_units)
    noise = rng.normal(size=n_units)
    return data.with_columns(
        pl.Series("w", np.repeat(weights, n_periods)), pl.Series("noise", np.repeat(noise, n_periods))
    )


@pytest.fixture
def wide_panel():
    """Generated panel of 1000 units over three periods with 20 covariates and a unit weight w and draw noise."""
    data = gen_did_scalable(1000, n_periods=3, n_cohorts=2, n_covariates=20, random_state=42)["data"]
    return _with_unit_draws(data, 1000, 3)


@pytest.fixture
def wider_panel():
    """Generated panel of 1000 units over three periods with 50 covariates and a unit weight w and draw noise."""
    data = gen_did_scalable(1000, n_periods=3, n_cohorts=2, n_covariates=50, random_state=42)["data"]
    return _with_unit_draws(data, 1000, 3)


@pytest.fixture
def wide_cross_section():
    """Generated cross sections of 400 rows in each of three periods with 20 covariates and row draws w and noise."""
    data = gen_did_scalable(400, n_periods=3, n_cohorts=2, n_covariates=20, panel=False, random_state=42)["data"]
    return _with_unit_draws(data, data.height, 1)


@pytest.fixture
def hub_overlap_panel():
    """Function that builds a two-period panel with 20 covariates whose first marks a hub of 600 treated units."""

    def build(n_hub_controls):
        rng = np.random.default_rng(0)
        cohort = np.repeat([2, 0, 2, 0], [600, n_hub_controls, 300, 600])
        n_units = cohort.size
        x = np.column_stack([np.r_[np.ones(600 + n_hub_controls), np.zeros(900)], rng.normal(size=(n_units, 19))])
        weights = rng.uniform(0.5, 1.5, n_units)
        y0 = x[:, :3] @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=n_units)
        y1 = y0 + x[:, :3] @ np.array([0.5, 0.2, -0.3]) + (cohort == 2) + rng.normal(size=n_units)
        covariates = {f"cov{j + 1}": x[:, j] for j in range(20)}
        return pl.concat(
            [
                pl.DataFrame({"id": np.arange(n_units), "time": t, "group": cohort, "y": y, "w": weights, **covariates})
                for t, y in ((1, y0), (2, y1))
            ]
        )

    return build


@pytest.fixture
def hub_overlap_cross_section():
    """Function that builds two cross sections with 20 covariates whose first marks a hub of 600 treated rows each."""

    def build(n_hub_controls):
        rng = np.random.default_rng(0)
        cohort = np.tile(np.repeat([2, 0, 2, 0], [600, n_hub_controls, 300, 600]), 2)
        time = np.repeat([1, 2], cohort.size // 2)
        hub = np.tile(np.r_[np.ones(600 + n_hub_controls), np.zeros(900)], 2)
        x = np.column_stack([hub, rng.normal(size=(cohort.size, 19))])
        weights = rng.uniform(0.5, 1.5, cohort.size)
        y = (
            x[:, :3] @ np.array([1.0, -1.0, 0.5])
            + (time == 2) * (x[:, :3] @ np.array([0.5, 0.2, -0.3]) + (cohort == 2))
            + rng.normal(size=cohort.size)
        )
        covariates = {f"cov{j + 1}": x[:, j] for j in range(20)}
        return pl.DataFrame(
            {"id": np.arange(cohort.size), "time": time, "group": cohort, "y": y, "w": weights, **covariates}
        )

    return build


@pytest.fixture
def newton_logit():
    """Function that fits a weighted logit by Newton steps and returns its scores clipped like the estimators."""

    def fit(d, x, weights):
        beta = np.zeros(x.shape[1])
        for _ in range(50):
            scores = 1 / (1 + np.exp(-x @ beta))
            hessian = x.T @ ((weights * scores * (1 - scores))[:, None] * x)
            beta = beta + np.linalg.lstsq(hessian, x.T @ (weights * (d - scores)), rcond=None)[0]
        return np.clip(1 / (1 + np.exp(-x @ beta)), 1e-6, 1 - 1e-6)

    return fit


@pytest.fixture
def reference_cell_att(newton_logit):
    """Function that computes the ATT of one cell of generated data from least squares fits and a Newton logit."""

    def fitted(y, x, weights, rows):
        root = np.sqrt(weights[rows])
        return x @ np.linalg.lstsq(x[rows] * root[:, None], y[rows] * root, rcond=None)[0]

    def mean(values, weights, rows):
        return np.sum(weights[rows] * values[rows]) / np.sum(weights[rows])

    def estimate(data, covariates, cell, est_method, panel=True, trim_level=0.995):
        group, later, base, comparison = cell
        in_cell = pl.col("group").is_in([group, *comparison])
        if panel:
            at = {period: data.filter((pl.col("time") == period) & in_cell).sort("id") for period in (later, base)}
            units = at[min(later, base)]
            y = at[later]["y"].to_numpy() - at[base]["y"].to_numpy()
        else:
            units = data.filter(pl.col("time").is_in([later, base]) & in_cell)
            y = units["y"].to_numpy()
            post = (units["time"] == later).to_numpy()
        d = (units["group"] == group).to_numpy()
        x = np.column_stack([np.ones(units.height), units.select(covariates).to_numpy()])
        weights = units["w"].to_numpy()
        scores = newton_logit(d.astype(float), x, weights)
        kept = weights * (d | (scores < trim_level))
        odds = kept * scores / (1 - scores)
        if panel:
            outcome = fitted(y, x, weights, ~d)
            if est_method == "reg":
                return mean(y - outcome, weights, d)
            change = y - outcome if est_method == "dr" else y
            return mean(change, kept, d) - mean(change, odds, ~d)
        sides = {(1, 1): d & post, (1, 0): d & ~post, (0, 1): ~d & post, (0, 0): ~d & ~post}
        if est_method == "ipw":
            treated = mean(y, kept, sides[1, 1]) - mean(y, kept, sides[1, 0])
            return treated - mean(y, odds, sides[0, 1]) + mean(y, odds, sides[0, 0])
        fits = {side: fitted(y, x, weights, rows) for side, rows in sides.items()}
        if est_method == "reg":
            treated = mean(y, weights, sides[1, 1]) - mean(y, weights, sides[1, 0])
            return treated - mean(fits[0, 1] - fits[0, 0], weights, d)
        residual = y - np.where(post, fits[0, 1], fits[0, 0])
        gaps = {period: fits[1, period] - fits[0, period] for period in (0, 1)}
        return (
            mean(residual, kept, sides[1, 1])
            - mean(residual, kept, sides[1, 0])
            - mean(residual, odds, sides[0, 1])
            + mean(residual, odds, sides[0, 0])
            + mean(gaps[1], kept, d)
            - mean(gaps[1], kept, sides[1, 1])
            - mean(gaps[0], kept, d)
            + mean(gaps[0], kept, sides[1, 0])
        )

    return estimate


@pytest.fixture
def wide_panel_atts_with_scaled_unit_weights():
    """Function that refits a generated panel after scaling the weight w of each unit by its own factor."""

    def refit(rows, units, factors, **spec):
        scaled = rows.with_columns(pl.col("w") * pl.col("id").replace_strict(units, factors))
        data = preprocess_did(scaled, yname="y", tname="time", idname="id", gname="group", weightsname="w", **spec)
        return np.array([cell.att for cell in compute_att_gt(data).attgt_list])

    return refit


@pytest.fixture
def wide_cross_section_atts_with_scaled_row_weights():
    """Function that refits generated cross sections after scaling the weight w of each row by its own factor."""

    def refit(rows, factors, **spec):
        scaled = rows.with_columns(pl.col("w") * pl.Series(factors))
        data = preprocess_did(scaled, yname="y", tname="time", gname="group", weightsname="w", panel=False, **spec)
        return np.array([cell.att for cell in compute_att_gt(data).attgt_list])

    return refit
