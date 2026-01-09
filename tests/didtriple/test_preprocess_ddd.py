"""Tests for DDD preprocessing."""

import pytest

from moderndid.core.preprocess.models import DDDData
from moderndid.core.preprocessing import preprocess_ddd_2periods
from moderndid.didtriple.dgp import gen_dgp_2periods

from ..helpers import importorskip

pd = importorskip("pandas")
np = importorskip("numpy")


def test_preprocess_ddd_basic():
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    with pytest.raises(ValueError, match="yname='missing' not found"):
        preprocess_ddd_2periods(
            data=data,
            yname="missing",
            tname="time",
            idname="id",
            gname="state",
            pname="partition",
        )


def test_preprocess_ddd_invalid_est_method():
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
    data = result["data"]

    extra = data[data["time"] == 1].copy()
    extra["time"] = 3
    data = pd.concat([data, extra], ignore_index=True)

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

    data = pd.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
        }
    )

    data.loc[(data["id"] == 0) & (data["time"] == 2), "partition"] = (
        1 - data.loc[(data["id"] == 0) & (data["time"] == 1), "partition"].values[0]
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

    data = pd.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "state": state,
            "partition": partition,
        }
    )

    data.loc[(data["id"] == 0) & (data["time"] == 2), "state"] = (
        1 - data.loc[(data["id"] == 0) & (data["time"] == 1), "state"].values[0]
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
    partition = np.repeat(np.array([1] * 1 + [0] * (n - 1)), 2)
    y = rng.normal(0, 1, n * 2)

    data = pd.DataFrame(
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
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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


@pytest.mark.parametrize("est_method", ["dr", "reg", "ipw"])
def test_preprocess_ddd_est_methods(est_method):
    result = gen_dgp_2periods(n=200, dgp_type=1, random_state=42)
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
