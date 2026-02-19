"""Shared fixtures for didtriple tests."""

import numpy as np
import polars as pl
import pytest

from moderndid import ddd_mp
from moderndid.core.preprocessing import preprocess_ddd_2periods
from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods
from moderndid.didtriple.estimators.ddd_mp import DDDMultiPeriodResult
from moderndid.didtriple.estimators.ddd_mp_rc import DDDMultiPeriodRCResult
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult
from moderndid.didtriple.estimators.ddd_rc import DDDRCResult


@pytest.fixture
def ddd_data_with_covariates():
    """Generate DDD data with covariates."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
    )

    covariates = np.column_stack([np.ones(ddd_data.n_units), ddd_data.covariates])

    return ddd_data, covariates


@pytest.fixture
def ddd_data_no_covariates():
    """Generate DDD data without covariates."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    covariates = np.ones((ddd_data.n_units, 1))

    return ddd_data, covariates


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
def two_period_df():
    """Raw 2-period DataFrame for ddd() wrapper tests."""
    dgp = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    return dgp["data"]


@pytest.fixture
def multi_period_df():
    """Raw multi-period DataFrame for ddd() wrapper tests."""
    dgp = gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    return dgp["data"]


@pytest.fixture
def two_period_dgp_result():
    """Full 2-period DGP result including true ATT and oracle ATT."""
    result = gen_dgp_2periods(n=1000, dgp_type=1, random_state=42)
    return result["data"], result["true_att"], result["oracle_att"]


def convert_r_array(arr):
    """Convert R array to numpy array."""
    result = []
    for val in arr:
        if val == "NA" or val is None:
            result.append(np.nan)
        else:
            result.append(float(val))
    return np.array(result)


@pytest.fixture
def bootstrap_data():
    """Bootstrap data for numba tests."""
    rng = np.random.default_rng(42)
    n = 100
    k = 5
    inf_func = rng.standard_normal((n, k))
    return inf_func


@pytest.fixture
def cluster_data():
    """Cluster data for numba tests."""
    rng = np.random.default_rng(42)
    n = 100
    k = 5
    inf_func = rng.standard_normal((n, k))
    cluster = np.repeat(np.arange(10), 10)
    return inf_func, cluster


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
def ddd_panel_result():
    return DDDPanelResult(
        att=2.5,
        se=0.5,
        uci=3.48,
        lci=1.52,
        boots=None,
        att_inf_func=np.random.default_rng(42).standard_normal(100),
        did_atts={"comparison_3": 1.0, "comparison_2": 0.8, "comparison_1": -0.7},
        subgroup_counts={
            "subgroup_4": 25,
            "subgroup_3": 25,
            "subgroup_2": 25,
            "subgroup_1": 25,
        },
        args={"est_method": "dr", "alpha": 0.05, "boot": False, "cband": False},
    )


@pytest.fixture
def ddd_panel_result_not_significant():
    return DDDPanelResult(
        att=0.1,
        se=0.5,
        uci=1.08,
        lci=-0.88,
        boots=None,
        att_inf_func=np.random.default_rng(42).standard_normal(100),
        did_atts={"comparison_3": 0.05, "comparison_2": 0.03, "comparison_1": -0.02},
        subgroup_counts={
            "subgroup_4": 25,
            "subgroup_3": 25,
            "subgroup_2": 25,
            "subgroup_1": 25,
        },
        args={"est_method": "dr", "alpha": 0.05, "boot": False},
    )


@pytest.fixture
def ddd_mp_result_fixture():
    rng = np.random.default_rng(42)
    return DDDMultiPeriodResult(
        att=np.array([1.5, 2.0, 2.5, 3.0]),
        se=np.array([0.3, 0.35, 0.4, 0.45]),
        lci=np.array([0.91, 1.31, 1.71, 2.11]),
        uci=np.array([2.09, 2.69, 3.29, 3.89]),
        groups=np.array([3, 3, 4, 4]),
        times=np.array([3, 4, 4, 5]),
        tlist=np.array([1, 2, 3, 4, 5]),
        glist=np.array([3, 4]),
        n=500,
        inf_func_mat=rng.standard_normal((500, 4)),
        unit_groups=np.repeat([0, 3, 4], [250, 125, 125]),
        args={
            "est_method": "dr",
            "alpha": 0.05,
            "control_group": "nevertreated",
            "base_period": "universal",
        },
    )


@pytest.fixture
def ddd_rc_result():
    rng = np.random.default_rng(42)
    return DDDRCResult(
        att=2.0,
        se=0.4,
        uci=2.78,
        lci=1.22,
        boots=None,
        att_inf_func=rng.standard_normal(200),
        did_atts={"comparison_3": 1.0, "comparison_2": 0.8, "comparison_1": -0.2},
        subgroup_counts={
            "subgroup_4": 50,
            "subgroup_3": 50,
            "subgroup_2": 50,
            "subgroup_1": 50,
        },
        args={"est_method": "dr", "alpha": 0.05, "boot": False},
    )


@pytest.fixture
def ddd_mp_rc_result():
    rng = np.random.default_rng(42)
    return DDDMultiPeriodRCResult(
        att=np.array([1.8, 2.2, 2.6, 3.0]),
        se=np.array([0.35, 0.38, 0.42, 0.48]),
        uci=np.array([2.49, 2.95, 3.43, 3.95]),
        lci=np.array([1.11, 1.45, 1.77, 2.05]),
        groups=np.array([3, 3, 4, 4]),
        times=np.array([3, 4, 4, 5]),
        glist=np.array([3, 4]),
        tlist=np.array([1, 2, 3, 4, 5]),
        inf_func_mat=rng.standard_normal((1500, 4)),
        n=1500,
        args={
            "est_method": "dr",
            "alpha": 0.05,
            "control_group": "nevertreated",
            "base_period": "universal",
        },
        unit_groups=rng.choice([0, 3, 4], size=1500),
    )


@pytest.fixture
def ddd_agg_result_simple():
    return DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="simple",
        egt=None,
        att_egt=None,
        se_egt=None,
        crit_val=1.96,
        inf_func=None,
        inf_func_overall=None,
        args={"alpha": 0.05, "boot": False, "cband": False},
    )


@pytest.fixture
def ddd_agg_result_eventstudy():
    return DDDAggResult(
        overall_att=2.0,
        overall_se=0.3,
        aggregation_type="eventstudy",
        egt=np.array([-2, -1, 0, 1, 2]),
        att_egt=np.array([0.1, 0.05, 1.8, 2.0, 2.2]),
        se_egt=np.array([0.2, 0.15, 0.3, 0.35, 0.4]),
        crit_val=1.96,
        inf_func=None,
        inf_func_overall=None,
        args={"alpha": 0.05, "boot": False, "cband": False},
    )


@pytest.fixture
def simple_panel_data():
    rng = np.random.default_rng(42)
    n_units = 100
    n_periods = 4

    records = []
    for unit in range(n_units):
        for t in range(1, n_periods + 1):
            records.append(
                {
                    "id": unit,
                    "time": t,
                    "y": rng.normal(0, 1),
                    "group": rng.choice([0, 3]),
                    "partition": rng.choice([0, 1]),
                }
            )

    return pl.DataFrame(records)


@pytest.fixture
def rcs_nuisance_data():
    rng = np.random.default_rng(42)
    n = 400

    post = np.repeat([0, 1], n // 2)
    subgroup = rng.choice([1, 2, 3, 4], size=n)
    y = rng.normal(0, 1, n) + 2.0 * (subgroup == 4) * post
    covariates = np.column_stack([np.ones(n), rng.normal(0, 1, (n, 2))])
    weights = np.ones(n)

    return y, post, subgroup, covariates, weights


@pytest.fixture
def ddd_baseline_result(two_period_df):
    from moderndid import ddd

    return ddd(
        data=two_period_df,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2",
        est_method="dr",
    )


@pytest.fixture
def ddd_converted(request, two_period_df):
    from tests.helpers import importorskip

    df_type = request.param
    if df_type == "pandas":
        importorskip("pandas")
        return two_period_df.to_pandas()
    if df_type == "pyarrow":
        importorskip("pyarrow")
        return two_period_df.to_arrow()
    if df_type == "duckdb":
        duckdb = importorskip("duckdb")
        conn = duckdb.connect()
        conn.register("ddd_data", two_period_df.to_arrow())
        return conn.execute("SELECT * FROM ddd_data").fetch_arrow_table()
    raise ValueError(f"Unknown dataframe type: {df_type}")
