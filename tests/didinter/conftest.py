"""Shared fixtures for didinter tests."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import load_favara_imbs
from moderndid.didinter.container import EffectsResult


@pytest.fixture(scope="module")
def favara_imbs_data():
    """Load the Favara and Imbs dataset."""
    return load_favara_imbs()


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def simple_panel_data(rng):
    """Simple panel data with known switchers at periods 3 and 4."""
    n_units = 50
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 20:
            switch_time = 3
            treatment[unit_mask & (periods >= switch_time)] = 1
        elif unit < 30:
            switch_time = 4
            treatment[unit_mask & (periods >= switch_time)] = 1

    y = rng.standard_normal(len(units))
    treatment_effect = 2.0
    y += treatment_effect * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.fixture
def bidirectional_panel_data(rng):
    """Panel data with units that switch treatment in both directions."""
    n_units = 30
    n_periods = 8

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 5:
            treatment[unit_mask & (periods >= 3) & (periods <= 5)] = 1
        elif unit < 15:
            treatment[unit_mask & (periods >= 4)] = 1

    y = rng.standard_normal(len(units)) + 1.5 * treatment

    return pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )


@pytest.fixture
def weighted_panel_data(simple_panel_data, rng):
    """Panel data with sampling weights."""
    weights = rng.uniform(0.5, 2.0, len(simple_panel_data))
    return simple_panel_data.with_columns(pl.Series("w", weights))


@pytest.fixture
def clustered_panel_data(simple_panel_data):
    """Panel data with cluster variable."""
    df = simple_panel_data.clone()
    cluster = (df["id"] // 10).cast(pl.Int64)
    return df.with_columns(cluster.alias("cluster"))


@pytest.fixture
def large_clustered_panel_data(rng):
    """Panel data with 100 units and 10 clusters for HC2BM tests."""
    n_units, n_periods = 100, 6
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(len(units))
    for unit in range(n_units):
        mask = units == unit
        if unit < 40:
            treatment[mask & (periods >= 3)] = 1
        elif unit < 60:
            treatment[mask & (periods >= 4)] = 1
    y = rng.standard_normal(len(units)) + 2.0 * treatment
    df = pl.DataFrame({"id": units, "time": periods, "y": y, "d": treatment})
    return df.with_columns((pl.col("id") // 10).cast(pl.Int64).alias("cluster"))


@pytest.fixture
def panel_with_controls(simple_panel_data, rng):
    """Panel data with control variables."""
    n = len(simple_panel_data)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    return simple_panel_data.with_columns(
        [
            pl.Series("x1", x1),
            pl.Series("x2", x2),
        ]
    )


@pytest.fixture
def unbalanced_panel_data(rng):
    """Unbalanced panel data with randomly missing observations."""
    n_units = 40
    n_periods = 6

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    treatment = np.zeros(len(units))
    for unit in range(n_units):
        unit_mask = units == unit
        if unit < 15:
            treatment[unit_mask & (periods >= 3)] = 1

    y = rng.standard_normal(len(units)) + 2.0 * treatment

    df = pl.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "d": treatment,
        }
    )

    keep_mask = rng.uniform(size=len(df)) > 0.15
    return df.filter(pl.Series(keep_mask))


@pytest.fixture
def basic_config():
    """Basic DIDInterConfig for testing."""
    from moderndid.core.preprocess.config import DIDInterConfig

    return DIDInterConfig(
        yname="y",
        tname="time",
        gname="id",
        dname="d",
    )


@pytest.fixture
def panel_data():
    """Small panel data with pre-computed switcher columns."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "y": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 2.0, 2.0, 2.0],
            "d": [0, 0, 1, 0, 1, 1, 0, 0, 0],
            "F_g": [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, float("inf"), float("inf"), float("inf")],
            "d_sq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "S_g": [1, 1, 1, 1, 1, 1, 0, 0, 0],
            "L_g": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, float("inf"), float("inf"), float("inf")],
        }
    )


@pytest.fixture
def switcher_data():
    """Panel data for testing delta_d computation."""
    return pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "time": [1, 2, 3, 1, 2, 3],
            "d": [0, 0, 1, 0, 1, 1],
            "d_sq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "F_g": [3.0, 3.0, 3.0, 2.0, 2.0, 2.0],
            "S_g": [1, 1, 1, 1, 1, 1],
            "weight_gt": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "dist_to_switch_1": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def ate_variance_inputs(rng):
    """Function that builds the influence functions of 100 groups and the cell counts for a number of horizons."""

    def build(n_horizons):
        counts = {f"count_{horizon}": rng.integers(0, 2, 30) for horizon in range(1, n_horizons + 1)}
        return {"influence_func_unnorm": rng.standard_normal((100, n_horizons)), "df": pl.DataFrame(counts)}

    return build


@pytest.fixture
def effects_results_basic(ate_variance_inputs):
    """Basic effects results dict for testing ATE computation."""
    return {
        "estimates": np.array([0.5, 0.6, 0.7]),
        "estimates_unnorm": np.array([0.5, 0.6, 0.7]),
        "n_switchers": np.array([100.0, 90.0, 80.0]),
        "n_switchers_weighted": np.array([100.0, 90.0, 80.0]),
        "ate_delta": np.array([1.0, 1.0, 1.0]),
        **ate_variance_inputs(3),
    }


@pytest.fixture
def effects_results_5():
    """Effects results dict with 5 horizons for range testing."""
    estimates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    vcov = np.eye(5) * 0.01 + 0.002
    return {"estimates": estimates, "vcov": vcov}


@pytest.fixture
def het_sample():
    """Small heterogeneity sample for regression tests."""
    rng = np.random.default_rng(42)
    n = 60
    return pl.DataFrame(
        {
            "_prod_het": rng.standard_normal(n),
            "weight_gt": rng.uniform(0.5, 2.0, n),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "F_g": np.repeat([3.0, 4.0, 5.0], n // 3),
            "d_sq": np.repeat([0.0, 1.0, 0.0], n // 3),
            "S_g": np.repeat([1.0, 1.0, -1.0], n // 3),
            "cluster_id": np.repeat(np.arange(n // 5), 5),
        }
    )


@pytest.fixture
def minimal_effects():
    """Minimal EffectsResult for result container tests."""
    return EffectsResult(
        horizons=np.array([1.0, 2.0]),
        estimates=np.array([0.5, 0.6]),
        std_errors=np.array([0.1, 0.12]),
        ci_lower=np.array([0.304, 0.365]),
        ci_upper=np.array([0.696, 0.835]),
        n_switchers=np.array([100.0, 90.0]),
        n_observations=np.array([500.0, 450.0]),
    )


@pytest.fixture
def hc2_config():
    """Config namespace for HC2 regression tests."""
    return SimpleNamespace(trends_nonparam=None, predict_het_hc2bm=False)


@pytest.fixture
def hc2bm_config():
    """Config namespace for HC2-BM clustered regression tests."""
    return SimpleNamespace(
        trends_nonparam=None,
        predict_het_hc2bm=True,
        cluster="cluster_id",
        gname="cluster_id",
    )


@pytest.fixture
def hc2bm_weighted_config():
    """Config namespace for HC2-BM clustered regression tests whose weights vary within clusters."""
    return SimpleNamespace(
        trends_nonparam=None,
        predict_het_hc2bm=True,
        cluster="cluster_id",
        gname="cluster_id",
        weightsname="weight_gt",
    )


@pytest.fixture
def block_hc2_by_hand():
    """Function that computes clustered HC2 standard errors from the general inverse square root of each block."""

    def compute(X, y, weights, clusters):
        bread = np.linalg.inv((X * weights[:, None]).T @ X)
        residuals = weights * (y - X @ (bread @ X.T @ (weights * y)))
        meat = np.zeros((X.shape[1], X.shape[1]))
        for cluster in np.unique(clusters):
            rows = clusters == cluster
            block = np.eye(rows.sum()) - X[rows] @ bread @ X[rows].T @ np.diag(weights[rows])
            score = X[rows].T @ (np.real(fractional_matrix_power(block, -0.5)) @ residuals[rows])
            meat += np.outer(score, score)
        return np.sqrt(np.diag(bread @ meat @ bread))

    return compute


@pytest.fixture
def variance_config():
    """Config namespace for the cohort and DOF helper tests."""
    return SimpleNamespace(tname="time", gname="id", trends_nonparam=None, less_conservative_se=False)


@pytest.fixture
def stepped_panel_data(rng):
    """Clustered panel where some switchers raise their treatment again one period after switching."""
    n_units, n_periods = 80, 7
    ids = np.arange(n_units)
    units = np.repeat(ids, n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    first_switch = np.where(ids < 30, 3, np.where(ids < 50, 4, 99))
    second_switch = np.where(ids % 3 == 0, first_switch + 1, 99)
    treatment = (periods >= first_switch[units]).astype(float) + (periods >= second_switch[units]).astype(float)
    y = rng.standard_normal(len(units)) + 1.5 * treatment
    return pl.DataFrame({"id": units, "time": periods, "y": y, "d": treatment, "cluster": units // 8})


@pytest.fixture
def fake_bootstrap():
    """Bootstrap result with known standard errors."""
    return SimpleNamespace(effects_se=np.array([0.5, 0.6]), placebos_se=None, ate_se=0.25)


@pytest.fixture
def draw_recorder():
    """Bootstrap estimate function that records each drawn panel."""
    frames = []

    def record(df, config):
        frames.append(df)
        return {"effects": np.full(config.effects, np.nan)}

    record.frames = frames
    return record


@pytest.fixture
def relabel_cluster_copies():
    """Function that stacks one copy of each drawn cluster and gives every copy new group ids."""

    def stack(data, draw, cluster, idname):
        copies = [data.filter(pl.col(cluster) == c).with_columns(pl.col(idname) + 1000 * k) for k, c in enumerate(draw)]
        return pl.concat(copies)

    return stack


@pytest.fixture
def two_way_panel_data(rng):
    """Panel whose groups with baseline treatment 1 raise or lower it at staggered dates."""
    n_periods = 6
    rows = []
    unit = 0
    for baseline, final, n_units, dates in [
        (0, 0, 10, None),
        (0, 1, 10, [4]),
        (0, 1, 6, [2]),
        (1, 1, 10, None),
        (1, 2, 12, [3, 4]),
        (1, 0, 12, [4, 5]),
    ]:
        for k in range(n_units):
            switch = dates[k % len(dates)] if dates else n_periods + 1
            effect = rng.normal()
            for t in range(1, n_periods + 1):
                dose = final if t >= switch else baseline
                y = effect + 0.3 * t + 1.5 * (dose - baseline) + rng.normal()
                rows.append((unit, t, float(dose), y, unit // 6))
            unit += 1
    return pl.DataFrame(rows, schema=["id", "time", "d", "y", "cluster"], orient="row")


@pytest.fixture
def all_switch_panel_data(rng):
    """Staggered panel in which every group eventually raises its treatment from 0 or lowers it from 1."""
    n_units, n_periods = 75, 6
    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)
    baseline = np.repeat([0.0, 0.0, 0.0, 1.0, 1.0], 15)[units]
    switched = periods >= np.repeat([3, 4, 5, 3, 4], 15)[units]
    treatment = np.where(switched, 1 - baseline, baseline)
    y = rng.standard_normal(len(units)) + 0.2 * periods + 1.5 * treatment
    return pl.DataFrame({"id": units, "time": periods, "y": y, "d": treatment})


@pytest.fixture
def first_effect_by_hand():
    """Function that computes the first effect from switchers and their not-yet-switched controls."""

    def compute(data, directions):
        df = data.sort(["id", "time"]).with_columns(
            (pl.col("y") - pl.col("y").shift(1).over("id")).alias("dy"),
            pl.col("d").first().over("id").alias("d1"),
        )
        switch = df.filter(pl.col("d") != pl.col("d1")).group_by("id").agg(pl.col("time").min().alias("f"))
        df = df.join(switch, on="id", how="left").with_columns(
            pl.col("f").fill_null(df["time"].max() + 1),
            pl.when(pl.col("d") > pl.col("d1")).then(1).otherwise(-1).alias("sign"),
        )
        gaps = []
        for row in df.filter((pl.col("time") == pl.col("f")) & pl.col("sign").is_in(directions)).iter_rows(named=True):
            controls = df.filter(
                (pl.col("d1") == row["d1"]) & (pl.col("time") == row["time"]) & (pl.col("f") > row["time"])
            )
            if controls.height > 0:
                gaps.append(row["sign"] * (row["dy"] - controls["dy"].mean()))
        return float(np.mean(gaps))

    return compute


@pytest.fixture
def cells_used_by_hand():
    """Function that counts the group-period cells that an effect or a placebo uses, each cell once."""

    def count(data, horizon, directions, placebo=False):
        lagged = pl.col("y").shift(horizon).over("id")
        change = pl.col("y").shift(2 * horizon).over("id") - lagged if placebo else pl.col("y") - lagged
        df = data.sort(["id", "time"]).with_columns(change.alias("dy"), pl.col("d").first().over("id").alias("d1"))
        switch = (
            df.filter(pl.col("d") != pl.col("d1"))
            .group_by("id")
            .agg(pl.col("time").min().alias("f"), pl.col("d").sort_by("time").first().alias("d_f"))
        )
        df = df.join(switch, on="id", how="left").with_columns(pl.col("f").fill_null(df["time"].max() + 1))
        sign = pl.when(pl.col("d_f") > pl.col("d1")).then(1).otherwise(-1)
        observed = pl.col("dy").is_not_null()
        controls = df.filter(observed & (pl.col("f") > pl.col("time")))
        switchers = df.filter(
            observed
            & pl.col("d_f").is_not_null()
            & (pl.col("time") == pl.col("f") - 1 + horizon)
            & sign.is_in(directions)
        ).join(controls.select("time", "d1").unique(), on=["time", "d1"], how="semi")
        used_controls = controls.join(switchers.select("time", "d1").unique(), on=["time", "d1"], how="semi")
        return pl.concat([switchers.select("id", "time"), used_controls.select("id", "time")]).unique().height

    return count


@pytest.fixture
def reach_panel():
    """Preprocessed panel whose switchers miss outcomes or controls at some horizons."""
    inf = float("inf")
    groups = {
        1: (3.0, 0.0, []),
        2: (3.0, 0.0, [4]),
        3: (4.0, 0.0, []),
        4: (4.0, 0.0, [1]),
        5: (inf, 0.0, []),
        6: (inf, 0.0, []),
        7: (3.0, 1.0, []),
        8: (5.0, 1.0, [4]),
    }
    last_period_with_controls = {0.0: 6.0, 1.0: 4.0}
    rows = []
    for unit, (switch, baseline, missing) in groups.items():
        for t in range(1, 7):
            y = None if t in missing else float(unit + t)
            weight = 0.0 if y is None else 1.0
            rows.append((unit, t, y, switch, last_period_with_controls[baseline], baseline, weight))
    return pl.DataFrame(rows, schema=["id", "time", "y", "F_g", "T_g", "d_sq", "weight_gt"], orient="row")


@pytest.fixture
def dose_panel_data(rng):
    """Panel whose switchers raise their treatment by 1 or 2 and whose larger raises miss their last outcome."""
    rows = []
    for unit, (switch, step) in enumerate([(None, 0)] * 6 + [(3, 1)] * 3 + [(3, 2)] * 3):
        for t in range(1, 5):
            dose = float(step) if switch is not None and t >= switch else 0.0
            y = rng.normal() + 0.5 * t + dose
            rows.append((unit, t, dose, None if step == 2 and t == 4 else y))
    return pl.DataFrame(rows, schema=["id", "time", "d", "y"], orient="row")


@pytest.fixture
def trends_panel_data(rng):
    """Panel with group trends and staggered switches whose missing outcomes change the switchers across horizons."""
    rows = []
    for unit in range(90):
        switch = [3, 4, 5, None][unit % 4]
        slope = rng.normal(scale=0.3)
        for t in range(1, 8):
            dose = float(switch is not None and t >= switch)
            rows.append((unit, t, dose, rng.normal() + slope * t + 1.5 * dose))
    df = pl.DataFrame(rows, schema=["id", "time", "d", "y"], orient="row")
    missing = pl.Series(rng.random(df.height) < 0.08) & (df["time"] > 2)
    return df.with_columns(pl.when(missing).then(None).otherwise(pl.col("y")).alias("y"))


@pytest.fixture
def first_differences():
    """Function that first-differences the outcome within groups and drops the first period."""

    def difference(data):
        return (
            data.sort(["id", "time"])
            .with_columns((pl.col("y") - pl.col("y").shift(1).over("id")).alias("y"))
            .filter(pl.col("time") > 1)
        )

    return difference


@pytest.fixture
def continuous_panel_data(rng):
    """Panel with distinct continuous baselines whose switchers raise or lower their treatment by one unit."""
    rows = []
    for unit in range(120):
        baseline = round(float(rng.uniform(0.5, 2.5)) * 64) / 64
        switch = [3, 4, 5, None][unit % 4]
        step = 1.0 if unit % 3 else -1.0
        for t in range(1, 7):
            dose = baseline + step * (switch is not None and t >= switch)
            rows.append((unit, t, dose, rng.normal() + 0.3 * baseline * t + (dose - baseline)))
    return pl.DataFrame(rows, schema=["id", "time", "d", "y"], orient="row")


@pytest.fixture
def baseline_trend_controls():
    """Function that moves every baseline to 0 and adds period-by-baseline polynomial covariates."""

    def build(data, degree):
        baseline = pl.col("d").sort_by("time").first().over("id")
        terms = [(t, k) for t in sorted(data["time"].unique().to_list())[1:] for k in range(1, degree + 1)]
        df = data.with_columns(
            (pl.col("d") - baseline).alias("d_change"),
            *[((pl.col("time") >= t).cast(pl.Float64) * baseline**k).alias(f"trend_{t}_{k}") for t, k in terms],
        )
        return df, "~ " + " + ".join(f"trend_{t}_{k}" for t, k in terms)

    return build


@pytest.fixture
def het_panel_data(rng):
    """Panel with one switching cohort, never-switchers, group trends, a group covariate, and group weights."""
    rows = []
    for unit in range(80):
        switch = 4 if unit < 50 else None
        x = float(rng.normal())
        weight = float(rng.uniform(0.5, 2.0))
        slope = rng.normal(scale=0.2)
        for t in range(1, 8):
            dose = float(switch is not None and t >= switch)
            rows.append((unit, t, dose, rng.normal(scale=0.5) + slope * t + (1.0 + 0.5 * x) * dose, x, weight))
    return pl.DataFrame(rows, schema=["id", "time", "d", "y", "x", "w"], orient="row")


@pytest.fixture
def switcher_changes():
    """Function that returns the covariate and the outcome change of each switcher in het_panel_data."""

    def changes(data, end, start=3):
        wide = data.filter(pl.col("id") < 50).pivot(on="time", index=["id", "x", "w"], values="y").sort("id")
        return wide, wide[str(end)] - wide[str(start)]

    return changes


@pytest.fixture
def trend_switchers_by_hand():
    """Function that counts the switchers whose outcome is observed from two periods before the switch to a horizon."""

    def count(data, horizon):
        switches = data.filter(pl.col("d") == 1).group_by("id").agg(pl.col("time").min().alias("f"))
        window = data.join(switches, on="id").filter(
            (pl.col("time") >= pl.col("f") - 2) & (pl.col("time") <= pl.col("f") - 1 + horizon)
        )
        observed = window.group_by("id").agg(pl.col("y").is_not_null().sum().alias("n"), pl.col("f").first())
        return observed.filter(pl.col("n") == horizon + 2).height

    return count
