# pylint: disable=redefined-outer-name
"""Shared fixtures for tests."""

import numpy as np
import pandas as pd
import pytest

from moderndid import att_gt, load_engel, load_mpdta
from moderndid.didcont.panel import PTEParams, process_att_gt


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n = 200
    w = np.random.uniform(0, 1, (n, 1))
    x = w + 0.2 * np.random.normal(0, 1, (n, 1))
    y = np.sin(2 * np.pi * x).ravel() + 0.1 * np.random.normal(0, 1, n)
    return y, x, w


@pytest.fixture
def multivariate_data():
    np.random.seed(42)
    n = 300
    w = np.random.uniform(0, 1, (n, 2))
    x = w @ np.array([[1.0, 0.5], [0.3, 0.8]]).T + 0.1 * np.random.normal(0, 1, (n, 2))
    y = np.sin(x[:, 0]) * np.cos(x[:, 1]) + 0.1 * np.random.normal(0, 1, n)
    return y, x, w


@pytest.fixture
def regression_data():
    np.random.seed(42)
    n = 150
    x = np.random.uniform(0, 1, (n, 1))
    y = 2 * x.ravel() + x.ravel() ** 2 + 0.1 * np.random.normal(0, 1, n)
    return y, x, x


@pytest.fixture
def continuous_data():
    np.random.seed(42)
    return np.random.normal(0, 1, (200, 3))


@pytest.fixture
def discrete_data():
    np.random.seed(42)
    n = 200
    return np.column_stack(
        [
            np.random.choice([0, 1, 2], n),
            np.random.choice([0, 1], n),
        ]
    )


@pytest.fixture
def degree_matrix():
    return np.array([[3, 3], [2, 2], [3, 4]])


@pytest.fixture
def indicator_vector():
    return np.array([1, 1])


@pytest.fixture
def simple_setup():
    np.random.seed(42)
    x = np.random.uniform(0, 1, (100, 2))
    K = np.array([[3, 4], [2, 3]])
    return x, K


@pytest.fixture
def basis_list():
    np.random.seed(42)
    return [
        np.random.normal(0, 1, (50, 3)),
        np.random.normal(0, 1, (50, 2)),
        np.random.normal(0, 1, (50, 4)),
    ]


@pytest.fixture
def simple_bases():
    return [
        np.array([[1, 2], [3, 4], [5, 6]]),
        np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]),
    ]


@pytest.fixture
def three_bases():
    return [
        np.array([[1, 2], [3, 4]]),
        np.array([[1], [2]]),
        np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
    ]


@pytest.fixture
def bspline_simple_data():
    return np.linspace(0, 1, 100)


@pytest.fixture
def random_uniform_data():
    np.random.seed(42)
    return np.random.uniform(0, 10, 200)


@pytest.fixture
def sparse_data():
    return np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])


@pytest.fixture
def selection_result():
    return {
        "j_x_seg": 3,
        "k_w_seg": 4,
        "j_tilde": 5,
        "theta_star": 1.2,
        "j_x_segments_set": np.array([2, 3, 4, 5]),
        "k_w_segments_set": np.array([3, 4, 5, 6]),
    }


@pytest.fixture
def matrices_small():
    np.random.seed(42)
    n = 50
    x = np.random.randn(n, 10)
    y = np.random.randn(n)
    y_pred = y + 0.1 * np.random.randn(n)
    symmetric_matrix = np.eye(5) + 0.1 * np.random.randn(5, 5)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2
    return x, y, y_pred, symmetric_matrix


@pytest.fixture
def matrices_large():
    np.random.seed(42)
    n = 5000
    p = 100
    x = np.random.randn(n, p)
    y = np.random.randn(n)
    y_pred = y + 0.1 * np.random.randn(n)
    symmetric_matrix = np.eye(50) + 0.1 * np.random.randn(50, 50)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix.T) / 2
    return x, y, y_pred, symmetric_matrix


@pytest.fixture
def basis_matrices():
    np.random.seed(42)
    n_obs = 100
    bases = [
        np.random.randn(n_obs, 3),
        np.random.randn(n_obs, 4),
        np.random.randn(n_obs, 2),
    ]
    return bases


@pytest.fixture
def engel_data():
    engel_df = load_engel()
    engel_df = engel_df.sort_values("logexp")

    return {
        "food": engel_df["food"].values,
        "logexp": engel_df["logexp"].values.reshape(-1, 1),
        "logwages": engel_df["logwages"].values.reshape(-1, 1),
    }


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_matrix(rng):
    return rng.randn(100, 5)


@pytest.fixture
def rank_deficient_matrix(rng):
    x = np.ones((100, 3))
    x[:, 1] = 2 * x[:, 0]
    x[:, 2] = rng.randn(100)
    return x


@pytest.fixture
def symmetric_psd_matrix(rng):
    A = rng.randn(5, 5)
    return A @ A.T


@pytest.fixture
def panel_data_balanced():
    np.random.seed(42)
    n_units = 20
    n_periods = 6
    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile(np.arange(2010, 2010 + n_periods), n_units)

    d_base = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    d_variation = 0.5 * np.random.normal(0, 1, len(unit_ids))
    d = np.maximum(0, d_base + d_variation)

    unit_effects = np.repeat(np.random.normal(0, 1, n_units), n_periods)
    time_effects = np.tile(np.random.normal(0, 0.5, n_periods), n_units)
    treatment_effect = 2.0 * d + 0.5 * d**2
    noise = np.random.normal(0, 1, len(unit_ids))
    y = unit_effects + time_effects + treatment_effect + noise

    x1 = np.random.normal(1, 0.5, len(unit_ids))
    x2 = np.random.binomial(1, 0.3, len(unit_ids))

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "time_id": time_ids,
            "y": y,
            "d": d,
            "x1": x1,
            "x2": x2,
            "weights": np.ones(len(unit_ids)),
        }
    )


@pytest.fixture
def panel_data_unbalanced():
    np.random.seed(42)
    n_units = 15
    max_periods = 8

    data_list = []
    for unit in range(1, n_units + 1):
        n_periods_unit = np.random.randint(3, max_periods + 1)
        start_year = np.random.randint(2008, 2012)
        periods = np.arange(start_year, start_year + n_periods_unit)

        unit_ids = np.full(n_periods_unit, unit)
        time_ids = periods

        d_base = np.random.normal(1, 0.8)
        d = np.maximum(0, d_base + 0.3 * np.random.normal(0, 1, n_periods_unit))

        unit_effect = np.random.normal(0, 1.2)
        time_effects = 0.1 * (periods - 2010)
        treatment_effect = 1.8 * d
        noise = np.random.normal(0, 0.8, n_periods_unit)
        y = unit_effect + time_effects + treatment_effect + noise

        x1 = np.random.normal(0, 1, n_periods_unit)
        x2 = np.random.exponential(1, n_periods_unit)
        weights = np.random.gamma(2, 0.5, n_periods_unit)

        unit_data = pd.DataFrame(
            {"unit_id": unit_ids, "time_id": time_ids, "y": y, "d": d, "x1": x1, "x2": x2, "weights": weights}
        )
        data_list.append(unit_data)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def staggered_treatment_panel():
    data = {
        "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "y": [10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38],
        "treat": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_simple_panel():
    data = {
        "id": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
        "time": [1, 2, 3, 1, 2, 3, 2, 3, 1, 2, 3],
        "y": [10, 12, 15, 20, 22, 25, 32, 35, 40, 42, 45],
        "treat": [0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def panel_data_with_covariates():
    np.random.seed(42)
    n_units = 25
    n_periods = 5
    n_obs = n_units * n_periods

    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile(np.arange(2015, 2015 + n_periods), n_units)

    age = np.repeat(np.random.randint(25, 65, n_units), n_periods)
    education = np.repeat(np.random.choice(["HS", "College", "Grad"], n_units), n_periods)
    income_base = np.repeat(np.random.lognormal(10, 0.8, n_units), n_periods)
    income = income_base + 0.05 * (time_ids - 2015) * income_base

    treatment_propensity = (
        0.02 * (age - 40) + 0.3 * (income / 50000) + 0.1 * (education == "College") + 0.2 * (education == "Grad")
    )
    d = np.maximum(0, treatment_propensity + 0.5 * np.random.normal(0, 1, n_obs))

    y = (
        2
        + 0.03 * age
        + 0.5 * (education == "College")
        + 0.8 * (education == "Grad")
        + 0.00002 * income
        + 1.5 * d
        - 0.1 * d**2
        + 0.2 * (time_ids - 2017)
        + np.random.normal(0, 1, n_obs)
    )

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "time_id": time_ids,
            "y": y,
            "d": d,
            "age": age,
            "education": education,
            "income": income,
            "weights": np.random.uniform(0.5, 1.5, n_obs),
        }
    )


@pytest.fixture
def panel_data_with_weights():
    np.random.seed(42)
    n_units = 30
    n_periods = 4
    n_obs = n_units * n_periods

    unit_ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    time_ids = np.tile([2018, 2019, 2020, 2021], n_units)

    stratum = np.repeat(np.random.choice(["A", "B", "C"], n_units, p=[0.5, 0.3, 0.2]), n_periods)
    base_weights = np.where(stratum == "A", 1.0, np.where(stratum == "B", 2.0, 3.0))

    weights = base_weights * np.random.uniform(0.8, 1.2, n_obs)

    treatment_multiplier = np.where(stratum == "A", 1.0, np.where(stratum == "B", 1.5, 0.8))
    d = treatment_multiplier * np.maximum(0, np.random.normal(1, 0.8, n_obs))

    stratum_effect = np.where(stratum == "A", 0, np.where(stratum == "B", 1.2, -0.8))
    y = stratum_effect + 2.0 * d + 0.1 * (time_ids - 2019) + np.random.normal(0, 1, n_obs)

    return pd.DataFrame(
        {"unit_id": unit_ids, "time_id": time_ids, "y": y, "d": d, "stratum": stratum, "weights": weights}
    )


@pytest.fixture
def mpdta():
    return load_mpdta()


@pytest.fixture
def att_gt_result(mpdta):
    return att_gt(
        data=mpdta,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        est_method="reg",
        bstrap=False,
    )


@pytest.fixture
def att_gt_result_bootstrap(mpdta):
    unique_counties = mpdta["countyreal"].unique()[:50]
    mpdta_subset = mpdta[mpdta["countyreal"].isin(unique_counties)]

    return att_gt(
        data=mpdta_subset,
        yname="lemp",
        tname="year",
        gname="first.treat",
        idname="countyreal",
        est_method="reg",
        bstrap=True,
        biters=100,
        cband=True,
    )


@pytest.fixture
def simple_influence_func():
    np.random.seed(42)
    n_obs = 100
    n_params = 5
    return np.random.randn(n_obs, n_params)


@pytest.fixture
def att_gt_raw_results():
    np.random.seed(42)
    n_groups = 3
    n_times = 4
    n_gt = n_groups * n_times

    attgt_list = []
    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006]:
            att_value = np.random.normal(0.1, 0.05) if g > t else np.random.normal(0, 0.02)
            attgt_list.append({"att": att_value, "group": g, "time_period": t})

    n_units = 50
    influence_func = np.random.randn(n_units, n_gt) * 0.1

    return {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": []}


@pytest.fixture
def pte_params_basic():
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(1, 51), 4),
            "time": np.tile([2003, 2004, 2005, 2006], 50),
            "y": np.random.randn(200),
            "G": np.repeat(np.random.choice([0, 2004, 2006, 2007], 50), 4),
        }
    )
    time_vals = data["time"].unique()
    time_map = {v: i + 1 for i, v in enumerate(sorted(time_vals))}
    data["period"] = data["time"].map(time_map).astype(int)
    data[".w"] = 1.0

    return PTEParams(
        yname="y",
        gname="G",
        tname="time",
        idname="id",
        data=data,
        g_list=np.array([2004, 2006, 2007]),
        t_list=np.array([2003, 2004, 2005, 2006]),
        cband=True,
        alp=0.05,
        boot_type="multiplier",
        anticipation=0,
        base_period="varying",
        weightsname=None,
        control_group="nevertreated",
        gt_type="att",
        ret_quantile=0.5,
        biters=100,
        cl=1,
        call="test_call",
        dname=None,
        degree=None,
        num_knots=None,
        knots=None,
        dvals=None,
        target_parameter=None,
        aggregation=None,
        treatment_type=None,
        xformula="~1",
    )


@pytest.fixture
def mock_att_gt_result(att_gt_raw_results, pte_params_basic):
    processed = process_att_gt(att_gt_raw_results, pte_params_basic)

    class MockATTGTResult:
        def __init__(self):
            self.groups = processed.groups
            self.times = processed.times
            self.att = processed.att
            self.influence_func = processed.influence_func
            self.pte_params = pte_params_basic

    return MockATTGTResult()


@pytest.fixture
def att_gt_result_with_data():
    np.random.seed(42)

    n_units = 100
    periods = [1, 2, 3, 4]
    groups = [0, 2, 3]

    unit_groups = np.random.choice(groups, n_units)
    data_list = []

    for period in periods:
        data_list.append(
            pd.DataFrame({"period": period, ".w": np.random.uniform(0.5, 2.0, n_units), "group": unit_groups})
        )

    data = pd.concat(data_list, ignore_index=True)

    att_values = []
    group_values = []
    time_values = []

    for g in [2, 3]:
        for t in periods:
            att_values.append(np.random.normal(0.1, 0.05))
            group_values.append(g)
            time_values.append(t)

    inf_func = np.random.randn(n_units, len(att_values)) * 0.1

    pte_params = PTEParams(
        yname="y",
        gname="group",
        tname="period",
        idname="id",
        data=data,
        g_list=np.array([2, 3]),
        t_list=np.array(periods),
        cband=True,
        alp=0.05,
        boot_type="multiplier",
        anticipation=0,
        base_period="varying",
        weightsname=None,
        control_group="nevertreated",
        gt_type="att",
        ret_quantile=0.5,
        biters=100,
        cl=1,
        call="test",
        dname=None,
        degree=None,
        num_knots=None,
        knots=None,
        dvals=None,
        target_parameter=None,
        aggregation=None,
        treatment_type=None,
        xformula="~1",
    )

    class MockATTGTResult:
        def __init__(self):
            self.groups = np.array(group_values)
            self.times = np.array(time_values)
            self.att = np.array(att_values)
            self.influence_func = inf_func
            self.pte_params = pte_params

    return MockATTGTResult()


@pytest.fixture
def dose_values():
    return np.linspace(0.1, 2.0, 20)


@pytest.fixture
def mock_pte_params_with_dose(pte_params_basic, dose_values):
    params_dict = pte_params_basic._asdict()
    params_dict.update(
        {
            "dname": "dose",
            "degree": 2,
            "num_knots": 2,
            "knots": np.array([0.7, 1.3]),
            "dvals": dose_values,
            "target_parameter": "ATT",
            "aggregation": "simple",
            "treatment_type": "continuous",
        }
    )
    return PTEParams(**params_dict)


@pytest.fixture
def mock_gt_results_with_dose():
    np.random.seed(42)
    n_groups = 3
    n_times = 4
    n_gt = n_groups * n_times
    n_units = 200
    n_doses = 20

    attgt_list = []
    extra_gt_returns = []

    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006]:
            att_value = np.random.normal(0.1, 0.05) if g > t else np.random.normal(0, 0.02)
            attgt_list.append({"att": att_value, "group": g, "time_period": t})

            n_basis = 5
            dose_results = {
                "group": g,
                "time_period": t,
                "extra_gt_returns": {
                    "att_dose": np.random.normal(0.1, 0.02, n_doses),
                    "acrt_dose": np.random.normal(0.05, 0.01, n_doses),
                    "att_overall": np.random.normal(0.1, 0.02),
                    "acrt_overall": np.random.normal(0.05, 0.01),
                    "beta": np.random.randn(n_basis),
                    "bread": np.random.randn(n_basis, n_basis),
                    "x_expanded": np.random.randn(50, n_basis),
                },
            }
            extra_gt_returns.append(dose_results)

    influence_func = np.random.randn(n_units, n_gt) * 0.1

    return {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": extra_gt_returns}


@pytest.fixture
def mock_gt_results_no_dose(att_gt_raw_results):
    return att_gt_raw_results


@pytest.fixture
def contdid_data():
    from tests.didcont.dgp import simulate_contdid_data

    data = simulate_contdid_data(n=1000, seed=12345)
    return data.rename(columns={"time_period": "period"})


@pytest.fixture
def cck_test_data():
    np.random.seed(42)
    n = 500

    id_vals = np.repeat(np.arange(1, n + 1), 2)
    time_vals = np.tile([1, 2], n)
    group_vals = np.repeat(np.random.choice([0, 1], n, p=[0.5, 0.5]), 2)

    pre_y = np.random.normal(0, 1, n)
    dose = np.where(group_vals[::2] == 1, np.random.uniform(0, 2, n), 0)
    post_y = pre_y + 2 * dose + np.random.normal(0, 1, n)

    y = np.empty(2 * n)
    y[::2] = pre_y
    y[1::2] = post_y

    d = np.repeat(dose, 2)

    return pd.DataFrame(
        {
            "id": id_vals,
            "time": time_vals,
            "y": y,
            "d": d,
            "g": group_vals,
        }
    )


@pytest.fixture
def panel_data_with_group(panel_data_balanced):
    data = panel_data_balanced.copy()

    def assign_group(unit):
        if unit <= 10:
            return 2011
        if unit <= 15:
            return 2012
        return 0

    data["group"] = data["unit_id"].apply(assign_group)
    data.loc[data["time_id"] < data["group"], "d"] = 0
    data.loc[data["group"] == 0, "d"] = 0
    return data


def create_mock_gt_results_with_correct_dimensions(degree, knots, n_doses=20):
    if knots is None:
        n_knots = 0
    else:
        n_knots = len(knots) if hasattr(knots, "__len__") else 0

    n_basis = 1 + degree + n_knots

    n_groups = 3
    n_times = 4
    n_gt = n_groups * n_times
    n_units = 200

    np.random.seed(42)
    attgt_list = []
    extra_gt_returns = []

    for g in [2004, 2006, 2007]:
        for t in [2003, 2004, 2005, 2006]:
            att_value = np.random.normal(0.1, 0.05) if g > t else np.random.normal(0, 0.02)
            attgt_list.append({"att": att_value, "group": g, "time_period": t})

            dose_results = {
                "group": g,
                "time_period": t,
                "extra_gt_returns": {
                    "att_dose": np.random.normal(0.1, 0.02, n_doses),
                    "acrt_dose": np.random.normal(0.05, 0.01, n_doses),
                    "att_overall": np.random.normal(0.1, 0.02),
                    "acrt_overall": np.random.normal(0.05, 0.01),
                    "beta": np.random.randn(n_basis),
                    "bread": np.random.randn(n_basis, n_basis),
                    "x_expanded": np.random.randn(50, n_basis),
                },
            }
            extra_gt_returns.append(dose_results)

    influence_func = np.random.randn(n_units, n_gt) * 0.1

    return {"attgt_list": attgt_list, "influence_func": influence_func, "extra_gt_returns": extra_gt_returns}


@pytest.fixture
def simple_panel_data():
    np.random.seed(42)
    n_units = 100

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)

    treatment = np.repeat(np.random.binomial(1, 0.5, n_units), 2)

    pre_outcome = np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pd.DataFrame({"id": id_vals, "D": treatment, "period": period_vals, "name": name_vals, "Y": y_vals})

    return df


@pytest.fixture
def panel_data_with_covariates_estimators():
    np.random.seed(42)
    n_units = 100

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)

    x1 = np.repeat(np.random.normal(0, 1, n_units), 2)
    x2 = np.repeat(np.random.normal(0, 1, n_units), 2)

    propensity = 1 / (1 + np.exp(-(x1[:n_units] + x2[:n_units])))
    treatment = np.repeat((np.random.uniform(size=n_units) < propensity).astype(int), 2)

    pre_outcome = x1[:n_units] + x2[:n_units] + np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pd.DataFrame(
        {"id": id_vals, "D": treatment, "period": period_vals, "name": name_vals, "Y": y_vals, "x1": x1, "x2": x2}
    )

    return df


@pytest.fixture
def balanced_panel_data_bootstrap():
    """Create balanced panel data for bootstrap testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": np.repeat([1, 2, 3], 4),
            "time": np.tile([2010, 2011, 2012, 2013], 3),
            "y": np.random.randn(12),
            "x": np.random.randn(12),
        }
    )


@pytest.fixture
def unbalanced_panel_data_bootstrap():
    """Create unbalanced panel data for bootstrap testing."""
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 1, 3, 1, 2, 3, 4],
            "y": np.random.randn(9),
        }
    )


@pytest.fixture
def basic_attgt_data():
    """Basic ATT(g,t) data for testing."""
    return [
        {"att": 0.1, "group": 2004, "time_period": 2003},
        {"att": 0.2, "group": 2004, "time_period": 2004},
        {"att": 0.15, "group": 2004, "time_period": 2005},
        {"att": 0.05, "group": 2006, "time_period": 2005},
        {"att": 0.12, "group": 2006, "time_period": 2006},
    ]


@pytest.fixture
def create_pte_params():
    """Factory for creating PTE parameters."""

    def _create(n_units=20, groups=None, times=None, gt_type="att", ret_quantile=0.5):
        if groups is None:
            groups = [2004, 2006]
        if times is None:
            times = [2003, 2004, 2005, 2006]

        data = pd.DataFrame(
            {
                "id": np.repeat(np.arange(1, n_units + 1), len(times)),
                "time": np.tile(times, n_units),
                "y": np.random.randn(n_units * len(times)),
                "G": np.repeat(np.random.choice([0] + groups, n_units), len(times)),
            }
        )

        return PTEParams(
            yname="y",
            gname="G",
            tname="time",
            idname="id",
            data=data,
            g_list=np.array(groups),
            t_list=np.array(times),
            cband=False,
            alp=0.05,
            boot_type="weighted",
            anticipation=0,
            base_period="varying",
            weightsname=None,
            control_group="nevertreated",
            gt_type=gt_type,
            ret_quantile=ret_quantile,
            biters=100,
            cl=1,
            call="test",
            dname=None,
            degree=None,
            num_knots=None,
            knots=None,
            dvals=None,
            target_parameter=None,
            aggregation=None,
            treatment_type=None,
            xformula="~1",
        )

    return _create


@pytest.fixture
def quantile_test_data():
    """Data for quantile aggregation tests."""
    np.random.seed(42)
    return {
        "attgt_list": [
            {"att": 0.1, "group": 2004, "time_period": 2003},
            {"att": 0.2, "group": 2004, "time_period": 2004},
        ],
        "extra_gt_returns_qtt": [
            {
                "group": 2004,
                "time_period": 2003,
                "extra_gt_returns": {
                    "F0": np.random.randn(100),
                    "F1": np.random.randn(100) + 0.5,
                },
            },
            {
                "group": 2004,
                "time_period": 2004,
                "extra_gt_returns": {
                    "F0": np.random.randn(100),
                    "F1": np.random.randn(100) + 0.3,
                },
            },
        ],
        "extra_gt_returns_qott": [
            {
                "group": 2004,
                "time_period": 2003,
                "extra_gt_returns": {"Fte": np.random.randn(100)},
            },
            {
                "group": 2004,
                "time_period": 2004,
                "extra_gt_returns": {"Fte": np.random.randn(100) + 0.2},
            },
        ],
    }


@pytest.fixture
def panel_data_with_groups():
    np.random.seed(42)
    n_units = 100

    id_vals = np.repeat(np.arange(n_units), 2)
    period_vals = np.tile([0, 1], n_units)
    name_vals = np.tile(["pre", "post"], n_units)

    group_vals = np.repeat(np.random.choice([2004, 2005, 2006], n_units), 2)

    treatment = np.repeat(np.random.binomial(1, 0.5, n_units), 2)

    pre_outcome = np.random.normal(0, 1, n_units)
    post_outcome = pre_outcome + 2 * treatment[:n_units] + np.random.normal(0, 1, n_units)
    y_vals = np.empty(2 * n_units)
    y_vals[::2] = pre_outcome
    y_vals[1::2] = post_outcome

    df = pd.DataFrame(
        {"id": id_vals, "D": treatment, "G": group_vals, "period": period_vals, "name": name_vals, "Y": y_vals}
    )

    return df
