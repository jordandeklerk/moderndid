import numpy as np
import pytest
import scipy.sparse as sp

from tests.helpers import importorskip

pl = importorskip("polars")
importorskip("moderndid.didml")

from moderndid import aggte_didml, blp_eventtimes, clan_ttest, didml, didml_agg, load_mpdta
from moderndid.didml import didml_result


@pytest.fixture
def mpdta_data():
    return load_mpdta()


@pytest.fixture
def mpdta_unbalanced(mpdta_data):
    return mpdta_data.filter(~((pl.col("countyreal") == 8001) & (pl.col("year") == 2007)))


@pytest.fixture
def mpdta_without_never_treated(mpdta_data):
    return mpdta_data.filter(pl.col("first.treat") != 0)


@pytest.fixture(scope="module")
def mpdta_short():
    """Keep the years 2005 to 2007 of mpdta without its 2004 cohort."""
    return load_mpdta().filter((pl.col("year") >= 2005) & (pl.col("first.treat") != 2004))


@pytest.fixture
def mpdta_short_unbalanced(mpdta_short):
    return mpdta_short.filter(~((pl.col("countyreal") == 8001) & (pl.col("year") == 2007)))


@pytest.fixture
def mpdta_short_without_county_8001(mpdta_short):
    return mpdta_short.filter(pl.col("countyreal") != 8001)


@pytest.fixture
def mpdta_2007_cohort(mpdta_data):
    """Keep the 2007 cohort and the never-treated counties of mpdta from 2004 on."""
    return mpdta_data.filter((pl.col("year") >= 2004) & pl.col("first.treat").is_in([0, 2007]))


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
    return {"k_folds": 2, "random_state": 0, "boot": False}


@pytest.fixture(scope="module")
def mpdta_analytic_fit():
    """Fit didml to mpdta with influence-function standard errors."""
    return didml(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        k_folds=2,
        random_state=0,
        boot=False,
    )


@pytest.fixture(scope="module")
def mpdta_bootstrap_fit():
    """Fit didml to mpdta with six bootstrap draws and a uniform band."""
    return didml(
        load_mpdta(),
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        k_folds=2,
        random_state=0,
        cband=True,
        biters=6,
    )


@pytest.fixture(scope="module")
def mpdta_short_bootstrap_fit(mpdta_short):
    """Fit didml to the short mpdta panel with two bootstrap draws."""
    return didml(
        mpdta_short,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
        k_folds=2,
        random_state=0,
        biters=2,
    )


@pytest.fixture
def mpdta_dynamic_agg(mpdta_analytic_fit):
    """Aggregate the analytic mpdta fit to event time."""
    return aggte_didml(mpdta_analytic_fit)


@pytest.fixture
def mpdta_group_agg(mpdta_dynamic_agg):
    """Relabel the dynamic aggregation of the mpdta fit as a group aggregation."""
    return mpdta_dynamic_agg._replace(aggregation_type="group")


@pytest.fixture
def mpdta_simple_agg(mpdta_dynamic_agg):
    """Keep only the overall effect of the dynamic aggregation of the mpdta fit."""
    return didml_agg(mpdta_dynamic_agg.overall_att, mpdta_dynamic_agg.overall_se, aggregation_type="simple")


@pytest.fixture
def mpdta_tiny_cohort(mpdta_data):
    """Keep the never-treated counties and the first three counties of the 2004 cohort in mpdta up to 2005."""
    keep = mpdta_data.filter(pl.col("first.treat") == 2004)["countyreal"].unique().sort()[:3].to_list()
    return mpdta_data.filter(
        (pl.col("year") <= 2005) & ((pl.col("first.treat") == 0) | pl.col("countyreal").is_in(keep))
    )


@pytest.fixture
def mpdta_with_state(mpdta_2007_cohort):
    """Add a state column taken from the leading digits of the county code to the 2007 cohort data."""
    return mpdta_2007_cohort.with_columns((pl.col("countyreal") // 1000).alias("state"))


@pytest.fixture
def mpdta_repeated_row(mpdta_data):
    return pl.concat([mpdta_data, mpdta_data.filter((pl.col("countyreal") == 17005) & (pl.col("year") == 2005))])


@pytest.fixture
def mpdta_short_infinite_outcome(mpdta_short):
    row = (pl.col("countyreal") == 8001) & (pl.col("year") == 2007)
    return mpdta_short.with_columns(pl.when(row).then(float("inf")).otherwise(pl.col("lemp")).alias("lemp"))


@pytest.fixture
def mpdta_zero_weights(mpdta_data):
    return mpdta_data.with_columns(pl.lit(0.0).alias("w"))


@pytest.fixture
def two_cohort_cells():
    """Build the cells of a hand-made two-cohort result over four units."""
    empty = sp.csr_matrix((4, 7))
    return {
        "groups": [2004.0, 2004.0, 2004.0, 2004.0, 2006.0, 2006.0, 2006.0],
        "times": [2004.0, 2005.0, 2006.0, 2007.0, 2005.0, 2006.0, 2007.0],
        "att_gt": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
        "se_gt": np.full(7, 0.1),
        "critical_value": 1.959963984540054,
        "influence_func": np.zeros((4, 7)),
        "cates": empty,
        "scores": empty,
        "gammas": empty,
        "unit_ids": [1, 2, 3, 4],
        "unit_periods": [0, 0, 0, 0],
        "n_units": 4,
        "estimation_params": {"cohort_counts": {2004.0: 3, 2006.0: 1}},
    }


@pytest.fixture
def two_cohort_result(two_cohort_cells):
    """Build the two-cohort result with the cohort of each unit."""
    return didml_result(**two_cohort_cells, G=[2004.0, 2004.0, 2004.0, 2006.0])


@pytest.fixture
def two_cohort_counts_only(two_cohort_cells):
    """Build the two-cohort result without unit cohorts."""
    return didml_result(**two_cohort_cells)


@pytest.fixture
def two_cohort_weighted(two_cohort_cells):
    """Build the two-cohort result with a sampling weight of 3 on the fourth unit."""
    return didml_result(**two_cohort_cells, G=[2004.0, 2004.0, 2004.0, 2006.0], weights_ind=[1.0, 1.0, 1.0, 3.0])


@pytest.fixture
def two_cohort_draws(two_cohort_cells):
    """Build the two-cohort result with three hand-made bootstrap draws."""
    draws = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
            [2.0, 2.0, 3.0, 4.0, 10.0, 22.0, 30.0],
            [1.0, 4.0, 3.0, 5.0, 12.0, 20.0, 34.0],
        ]
    )
    shares = np.array([[0.75, 0.25], [0.5, 0.5], [0.6, 0.2]])
    return didml_result(
        **two_cohort_cells,
        G=[2004.0, 2004.0, 2004.0, 2006.0],
        att_gt_draws=draws,
        cohort_share_draws=shares,
    )


@pytest.fixture
def two_cohort_draws_missing_a_cohort(two_cohort_cells):
    """Build the two-cohort result with a fourth draw that holds no unit of the 2004 cohort."""
    nan = np.nan
    draws = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
            [2.0, 2.0, 3.0, 4.0, 10.0, 22.0, 30.0],
            [1.0, 4.0, 3.0, 5.0, 12.0, 20.0, 34.0],
            [nan, nan, nan, nan, 11.0, 21.0, 31.0],
        ]
    )
    shares = np.array([[0.75, 0.25], [0.5, 0.5], [0.6, 0.2], [0.0, 0.4]])
    return didml_result(
        **two_cohort_cells,
        G=[2004.0, 2004.0, 2004.0, 2006.0],
        att_gt_draws=draws,
        cohort_share_draws=shares,
    )


@pytest.fixture
def het_data():
    rng = np.random.default_rng(7)
    cates = np.tile(np.arange(20, dtype=float), 2)
    return pl.DataFrame(
        {
            "event_time": np.repeat([0, 1], 20),
            "dynamic_cate": cates,
            "x": 0.1 * cates + rng.standard_normal(40),
        }
    )


@pytest.fixture
def het_data_padded(het_data):
    first = het_data.filter(pl.col("event_time") == 0)
    return pl.concat(
        [
            first.with_columns(pl.lit(-1, dtype=pl.Int64).alias("event_time")),
            het_data,
            first.with_columns(
                pl.lit(2, dtype=pl.Int64).alias("event_time"),
                pl.lit(None, dtype=pl.Float64).alias("dynamic_cate"),
            ),
        ]
    )


@pytest.fixture
def het_blp(het_data):
    """Fit the best linear predictor of the heterogeneity data on x at event times 0 and 1."""
    return blp_eventtimes(het_data, n_periods=1, rhs_formula="x")


@pytest.fixture
def het_clan(het_data):
    """Compare the top and bottom quintiles of the heterogeneity data on x and dynamic_cate at event time 0."""
    return clan_ttest(het_data, affected=["x", "dynamic_cate"], threshold=0.2)[0]


@pytest.fixture
def het_panel():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [2004, 2005, 2006] * 3,
            "group": [2005] * 6 + [0] * 3,
            "x": [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5],
        }
    )


@pytest.fixture
def het_cates():
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "event_time": [-1, 0, 1, -1, 0, 1],
            "cate": [0.0, 0.3, 0.4, 0.1, 0.5, 0.6],
        }
    )


@pytest.fixture
def stacked_cell():
    rng = np.random.default_rng(3)
    n = 80
    covariates = rng.standard_normal((n, 2))
    cohort = (rng.uniform(size=n) < 0.5).astype(float)
    y_pre = covariates @ [0.5, -0.3] + rng.standard_normal(n)
    y_post = y_pre + 0.2 + cohort * (1.0 + 0.5 * covariates[:, 0]) + rng.standard_normal(n)
    return {
        "X": np.vstack([covariates, covariates]),
        "Y": np.concatenate([y_pre, y_post]),
        "post_indicator": np.concatenate([np.zeros(n), np.ones(n)]),
        "cohort_indicator": np.concatenate([cohort, cohort]),
    }
