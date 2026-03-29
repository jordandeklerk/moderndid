"""Tests for the main dynamic covariate balancing estimator function."""

import numpy as np
import polars as pl
import pytest

import moderndid.dev.diddynamic.format  # noqa: F401
from moderndid.dev.diddynamic.container import DynBalancingResult
from moderndid.dev.diddynamic.dyn_balancing import dyn_balancing


def test_returns_result(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert isinstance(result, DynBalancingResult)


def test_att_is_finite(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert np.isfinite(result.att)


def test_se_is_positive(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.se > 0


def test_contains_expected_keys(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    params = result.estimation_params
    assert params["yname"] == "y"
    assert params["balancing"] == "dcb"
    assert params["n_units"] == 60
    assert params["ds1"] == [0, 1, 1]
    assert params["ds2"] == [0, 0, 0]


def test_missing_treatment_name_raises(estimator_panel):
    with pytest.raises((ValueError, TypeError)):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
        )


def test_invalid_alp_raises(estimator_panel):
    with pytest.raises(ValueError, match="alp must be between"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            alp=1.5,
        )


def test_invalid_balancing_raises(estimator_panel):
    with pytest.raises(ValueError, match="balancing must be one of"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            balancing="invalid",
        )


def test_invalid_method_raises(estimator_panel):
    with pytest.raises(ValueError, match="method must be one of"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            method="invalid",
        )


def test_ds_length_mismatch_raises(estimator_panel):
    with pytest.raises(ValueError, match="same length"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1],
            ds2=[0, 0, 0],
        )


def test_empty_ds1_raises(estimator_panel):
    with pytest.raises(ValueError, match="ds1 must be a non-empty"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[],
            ds2=[0, 0, 0],
        )


def test_empty_ds2_raises(estimator_panel):
    with pytest.raises(ValueError, match="ds2 must be a non-empty"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[],
        )


@pytest.mark.parametrize("alp", [0.0, 1.0, -0.1, 2.0])
def test_boundary_alp_raises(estimator_panel, alp):
    with pytest.raises(ValueError, match="alp must be between"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            alp=alp,
        )


def test_lb_greater_than_ub_raises(estimator_panel):
    with pytest.raises(ValueError, match="lb.*must be less than"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            lb=10.0,
            ub=0.1,
        )


def test_continuous_treatment_with_regularization_raises(estimator_panel):
    with pytest.raises(ValueError, match="Regularization with continuous treatment"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            continuous_treatment=True,
            regularization=True,
        )


def test_large_alpha_warns(estimator_panel):
    with pytest.warns(UserWarning, match="Significance level larger than 0.1"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla="~ X1",
            alp=0.2,
            ub=20.0,
            grid_length=50,
            nfolds=3,
            adaptive_balancing=False,
        )


def test_single_period_ds_warns(estimator_panel):
    with pytest.warns(UserWarning, match="No dynamics"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1],
            ds2=[0],
            xformla="~ X1",
            ub=20.0,
            grid_length=50,
            nfolds=3,
            adaptive_balancing=False,
        )


def test_pooled_auto_sets_cluster(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        pooled=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.estimation_params.get("clustervars") == ["id"]


def test_invalid_final_period_raises(estimator_panel):
    with pytest.raises(ValueError, match="final_period.*not in the data"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1, 1],
            ds2=[0, 0],
            final_period=999,
        )


def test_treatment_history_too_long_raises(estimator_panel):
    with pytest.raises(ValueError, match="not in the data"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1] * 10,
            ds2=[0] * 10,
        )


def test_with_covariates(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1 + X2",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert isinstance(result, DynBalancingResult)
    assert np.isfinite(result.att)


def test_with_fixed_effects(estimator_panel):
    panel = estimator_panel.with_columns((pl.col("id") % 3).alias("fe_group"))
    result = dyn_balancing(
        data=panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        fixed_effects=["fe_group"],
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert isinstance(result, DynBalancingResult)
    assert np.isfinite(result.att)


def test_se_positive(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        clustervars=["cluster_var"],
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert isinstance(result, DynBalancingResult)
    assert result.se > 0


@pytest.mark.parametrize("bal_method", ["ipw", "aipw", "ipw_msm"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_ipw_returns_result(estimator_panel, bal_method):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        balancing=bal_method,
        nfolds=3,
    )
    assert isinstance(result, DynBalancingResult)
    assert np.isfinite(result.att)
    assert result.se > 0


def test_str_contains_title(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    output = str(result)
    assert "Dynamic Covariate Balancing" in output
    assert "ATE" in output


def test_recovers_known_effect():
    rng = np.random.default_rng(123)
    n_units = 100
    n_periods = 2
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    x1 = np.repeat(rng.standard_normal(n_units), n_periods)
    treatment = np.zeros(n_units * n_periods)
    for i in range(n_units // 2):
        treatment[i * n_periods + 1] = 1.0
    true_ate = 2.0
    y = np.repeat(rng.standard_normal(n_units), n_periods)
    for i in range(n_units * n_periods):
        if treatment[i] == 1.0:
            y[i] += true_ate
    df = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "D": treatment,
            "X1": x1,
        }
    )
    result = dyn_balancing(
        data=df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1],
        ds2=[0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.att == pytest.approx(true_ate, abs=2.0)


def test_zero_effect_with_no_treatment():
    rng = np.random.default_rng(456)
    n_units = 80
    n_periods = 2
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    treatment = np.zeros(n_units * n_periods)
    for i in range(n_units // 2):
        treatment[i * n_periods + 1] = 1.0
    y = np.repeat(rng.standard_normal(n_units), n_periods) + rng.standard_normal(n_units * n_periods) * 0.1
    x1 = np.repeat(rng.standard_normal(n_units), n_periods)
    df = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": y,
            "D": treatment,
            "X1": x1,
        }
    )
    result = dyn_balancing(
        data=df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1],
        ds2=[0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.att == pytest.approx(0.0, abs=2.0)
