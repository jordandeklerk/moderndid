"""Tests for DDD preprocessing."""

import re
import warnings

import pytest

from moderndid.core.preprocess.builders import PreprocessDataBuilder
from moderndid.core.preprocess.config import DDDConfig
from moderndid.core.preprocess.models import DDDData
from moderndid.core.preprocess.utils import check_partition_collinearity
from moderndid.core.preprocessing import preprocess_ddd_2periods
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_ddd_2periods
from tests.helpers import importorskip

np = importorskip("numpy")
pl = importorskip("polars")


def test_preprocess_ddd_basic():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    assert isinstance(ddd_data, DDDData)
    assert ddd_data.n_units == 200
    assert len(ddd_data.y0) == 200
    assert len(ddd_data.y1) == 200
    assert len(ddd_data.treat) == 200
    assert len(ddd_data.partition) == 200
    assert len(ddd_data.subgroup) == 200
    assert len(ddd_data.weights) == 200


def test_preprocess_ddd_with_covariates():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
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

    assert ddd_data.has_covariates
    assert ddd_data.covariates.shape == (200, 4)
    assert ddd_data.covariate_names == ["cov1", "cov2", "cov3", "cov4"]


def test_preprocess_ddd_no_covariates():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    assert not ddd_data.has_covariates
    assert ddd_data.covariates.shape[1] == 0


def test_preprocess_ddd_subgroup_assignment():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    for i in range(ddd_data.n_units):
        treat = ddd_data.treat[i]
        part = ddd_data.partition[i]
        sg = ddd_data.subgroup[i]

        if treat == 1 and part == 1:
            assert sg == 4
        elif treat == 1 and part == 0:
            assert sg == 3
        elif treat == 0 and part == 1:
            assert sg == 2
        else:
            assert sg == 1


def test_preprocess_ddd_subgroup_counts():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    assert len(ddd_data.subgroup_counts) == 4
    assert all(count >= 5 for count in ddd_data.subgroup_counts.values())
    assert sum(ddd_data.subgroup_counts.values()) == ddd_data.n_units


def test_preprocess_ddd_weight_normalization():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    assert np.isclose(ddd_data.weights.mean(), 1.0)


def test_preprocess_ddd_config_stored():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method="dr",
        alp=0.05,
    )

    assert ddd_data.config.yname == "y"
    assert ddd_data.config.tname == "time"
    assert ddd_data.config.idname == "id"
    assert ddd_data.config.gname == "state"
    assert ddd_data.config.pname == "partition"
    assert ddd_data.config.alp == 0.05


def test_preprocess_ddd_missing_column():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    with pytest.raises(ValueError, match="yname='missing' is not a column in the data"):
        preprocess_ddd_2periods(
            data=data,
            yname="missing",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_invalid_est_method():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    with pytest.raises(ValueError, match="est_method must be"):
        preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            est_method="invalid",
        )


def test_preprocess_ddd_more_than_two_periods():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    extra = data.filter(pl.col("time") == 1).with_columns(pl.lit(3).cast(pl.Int64).alias("time"))
    data = pl.concat([data, extra])

    with pytest.raises(ValueError, match="exactly 2 time periods"):
        preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_partition_time_varying():
    rng = np.random.RandomState(42)
    n = 100
    ids = np.repeat(np.arange(n), 2)
    times = np.tile([1, 2], n)
    state = np.repeat(rng.binomial(1, 0.5, n), 2)
    partition = np.repeat(rng.binomial(1, 0.5, n), 2)
    y = rng.normal(0, 1, n * 2)

    data = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
        }
    )

    original_val = data.filter((pl.col("id") == 0) & (pl.col("time") == 1))["partition"][0]
    new_val = 1 - original_val
    data = data.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("time") == 2))
        .then(new_val)
        .otherwise(pl.col("partition"))
        .alias("partition")
    )

    with pytest.raises(ValueError, match="partition.*same across all periods"):
        preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_treatment_time_varying():
    rng = np.random.RandomState(42)
    n = 100
    ids = np.repeat(np.arange(n), 2)
    times = np.tile([1, 2], n)
    state = np.repeat(rng.binomial(1, 0.5, n), 2)
    partition = np.repeat(rng.binomial(1, 0.5, n), 2)
    y = rng.normal(0, 1, n * 2)

    data = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
        }
    )

    original_val = data.filter((pl.col("id") == 0) & (pl.col("time") == 1))["state"][0]
    new_val = 1 - original_val
    data = data.with_columns(
        pl.when((pl.col("id") == 0) & (pl.col("time") == 2)).then(new_val).otherwise(pl.col("state")).alias("state")
    )

    with pytest.raises(ValueError, match="state.*same across all periods"):
        preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_small_subgroup():
    rng = np.random.RandomState(42)
    n = 20
    ids = np.repeat(np.arange(n), 2)
    times = np.tile([1, 2], n)
    state = np.repeat(np.array([1] * 2 + [0] * (n - 2)), 2)
    partition = np.repeat(np.array([1, 0] + [1] * 9 + [0] * 9), 2)
    y = rng.normal(0, 1, n * 2)

    data = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
        }
    )

    with pytest.raises(ValueError, match="Subgroup.*only.*observations"):
        preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_with_cluster():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        cluster="cluster",
        boot=True,
    )

    assert ddd_data.has_cluster
    assert len(ddd_data.cluster) == ddd_data.n_units


def test_preprocess_ddd_stores_inference_options_as_given():
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)

    ddd_data = preprocess_ddd_2periods(
        data=result["data"],
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        cluster="cluster",
        alp=0.2,
    )

    assert ddd_data.config.alp == 0.2
    assert ddd_data.config.boot is False
    assert ddd_data.config.cband is False


def test_preprocess_ddd_pairs_outcomes_by_unit_id(two_period_df):
    shuffled = two_period_df.sample(fraction=1.0, shuffle=True, seed=3)
    wide = two_period_df.pivot(on="time", index="id", values="y").sort("id")

    ddd_data = preprocess_ddd_2periods(
        data=shuffled,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
    )

    np.testing.assert_array_equal(ddd_data.y0, wide["1"].to_numpy())
    np.testing.assert_array_equal(ddd_data.y1, wide["2"].to_numpy())


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_ddd_checks_the_partition_on_the_rows_that_remain(two_period_df, missing):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "state", "pname": "partition"}
    data = two_period_df.with_columns(pl.col("partition").cast(pl.Float64))
    row = (pl.col("id") == 1) & (pl.col("time") == 2)
    expected = ddd(data.filter(pl.col("id") != 1), **spec)

    with pytest.warns(UserWarning, match="^Dropped 1 units while converting to balanced panel$"):
        result = ddd(
            data.with_columns(pl.when(row).then(missing).otherwise(pl.col("partition")).alias("partition")), **spec
        )

    assert result.att == expected.att
    assert result.se == expected.se


@pytest.mark.parametrize(("data_name", "gname"), [("two_period_df", "state"), ("multi_period_df", "group")])
@pytest.mark.parametrize(
    ("argument", "value", "suggestion"),
    [("tname", "tim", "time"), ("idname", "idd", "id"), ("pname", "partitoin", "partition")],
)
def test_ddd_names_misspelled_columns(request, data_name, gname, argument, value, suggestion):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": gname, "pname": "partition"} | {argument: value}
    message = f"{argument}='{value}' is not a column in the data. Did you mean '{suggestion}'?"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        ddd(request.getfixturevalue(data_name), **spec)


@pytest.mark.parametrize("two_period_duplicated_df", ["compensating_rows"], indirect=True)
def test_ddd_builder_drops_units_whose_repeated_rows_hide_a_missing_period(two_period_duplicated_df, two_period_df):
    config = DDDConfig(yname="y", tname="time", idname="id", gname="state", pname="partition")

    with pytest.warns(UserWarning, match="Dropped 2 units while converting to balanced panel"):
        result = PreprocessDataBuilder().with_data(two_period_duplicated_df).with_config(config).transform().build()

    clean = two_period_df.filter(~pl.col("id").is_in([11, 21]))
    expected = PreprocessDataBuilder().with_data(clean).with_config(config).transform().build()
    np.testing.assert_array_equal(result.y1, expected.y1)
    np.testing.assert_array_equal(result.y0, expected.y0)
    np.testing.assert_array_equal(result.subgroup, expected.subgroup)


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_preprocess_ddd_est_methods(est_method):
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    ddd_data = preprocess_ddd_2periods(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        est_method=est_method,
    )

    assert ddd_data.config.est_method.value == est_method


def test_partition_collinearity_detection():
    n = 100
    subgroup = np.array([4] * 25 + [3] * 25 + [2] * 25 + [1] * 25)
    x1 = np.random.default_rng(42).standard_normal(n)
    x2 = np.random.default_rng(42).standard_normal(n)
    x3 = np.random.default_rng(42).standard_normal(n)
    x3[:50] = x1[:50]

    cov_matrix = np.column_stack([x1, x2, x3])
    var_names = ["x1", "x2", "x3"]

    partition_collinear, all_collinear = check_partition_collinearity(cov_matrix, subgroup, var_names)

    assert "x3" in all_collinear
    assert "x3" in partition_collinear
    assert "subgroup 4 vs 3" in partition_collinear["x3"]


def test_partition_collinearity_warning_in_preprocess():
    rng = np.random.default_rng(42)
    n = 200
    ids = np.repeat(np.arange(n), 2)
    times = np.tile([1, 2], n)
    state = np.repeat(rng.choice([0, 2], n, p=[0.5, 0.5]), 2)
    partition = np.repeat(rng.choice([0, 1], n, p=[0.5, 0.5]), 2)
    cov1 = np.repeat(rng.standard_normal(n), 2)
    cov2 = np.repeat(rng.standard_normal(n), 2)
    y = rng.standard_normal(n * 2)

    data = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
            "cov1": cov1,
            "cov2": cov2,
        }
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ddd_data = preprocess_ddd_2periods(
            data=data,
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            xformla="~ cov1 + cov2",
        )

    assert ddd_data.covariates.shape[1] <= 2


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"yname": "y", "tname": "time", "pname": "partition"}, "gname is required"),
        ({"yname": "y", "tname": "time", "gname": "state"}, "pname is required"),
    ],
)
def test_ddd_missing_required_params(kwargs, match):
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    with pytest.raises(ValueError, match=match):
        ddd(data=result["data"], **kwargs)


@pytest.mark.parametrize("est_method", ["ols", "lasso", ""])
def test_ddd_invalid_est_method(est_method):
    result = gen_ddd_2periods(n=200, dgp_type=1, random_state=42)
    with pytest.raises(ValueError, match=f"est_method='{est_method}' is not valid"):
        ddd(
            data=result["data"],
            yname="y",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
            est_method=est_method,
        )
