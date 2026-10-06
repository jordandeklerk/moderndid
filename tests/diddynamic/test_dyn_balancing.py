"""Tests for the main dynamic covariate balancing estimator function."""

import numpy as np
import polars as pl
import pytest
from scipy.stats import chi2

import moderndid.diddynamic.format  # noqa: F401
from moderndid.core.converters import dynbalancinghetresult_to_polars, dynbalancingresult_to_polars
from moderndid.diddynamic.container import DynBalancingResult
from moderndid.diddynamic.dyn_balancing import dyn_balancing


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


def test_continuous_treatment_raises_not_implemented(estimator_panel):
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            continuous_treatment=True,
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


@pytest.mark.filterwarnings("ignore:pooled=True has no effect:UserWarning")
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


@pytest.mark.parametrize("bal_method", ["ipw", "aipw"])
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


def test_history_summary_matches_individual_results(history_result):
    for i, row in enumerate(history_result.summary.iter_rows(named=True)):
        r = history_result.results[i]
        assert row["att"] == r.att
        assert row["var_att"] == r.var_att
        assert row["mu1"] == r.mu1
        assert row["mu2"] == r.mu2
        assert row["var_mu1"] == r.var_mu1
        assert row["var_mu2"] == r.var_mu2
        assert row["robust_quantile"] == r.robust_quantile
        assert row["gaussian_quantile"] == r.gaussian_quantile


def test_history_period_lengths_sorted(history_result):
    assert history_result.summary["period_length"].to_list() == [1, 2, 3]


def test_history_att_equals_mu1_minus_mu2(history_result):
    for row in history_result.summary.iter_rows(named=True):
        assert row["att"] == pytest.approx(row["mu1"] - row["mu2"], abs=1e-10)


def test_history_var_att_equals_var_sum(history_result):
    for row in history_result.summary.iter_rows(named=True):
        assert row["var_att"] == pytest.approx(row["var_mu1"] + row["var_mu2"], abs=1e-10)


@pytest.mark.filterwarnings("ignore:ds1 contains one element:UserWarning")
def test_history_slices_ds_correctly(estimator_panel):
    ds1 = [0, 1, 1]
    ds2 = [0, 0, 0]
    hist = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=ds1,
        ds2=ds2,
        histories_length=[1, 3],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    single = dyn_balancing(
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
    assert hist.results[0].att == pytest.approx(single.att, abs=1e-10)


@pytest.mark.parametrize(
    "histories_length, match",
    [
        ([], "non-empty"),
        ([0, 2], "between 1 and"),
        ([4], "between 1 and"),
    ],
)
def test_history_invalid_lengths_raise(estimator_panel, histories_length, match):
    with pytest.raises(ValueError, match=match):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            histories_length=histories_length,
            xformla="~ X1",
        )


def test_history_repr_contains_table(history_result):
    text = str(history_result)
    assert "ATE" in text
    assert "Length" in text


def test_het_summary_matches_individual_results(het_result):
    for i, row in enumerate(het_result.summary.iter_rows(named=True)):
        r = het_result.results[i]
        assert row["att"] == r.att
        assert row["var_att"] == r.var_att
        assert row["mu1"] == r.mu1
        assert row["mu2"] == r.mu2
        assert row["var_mu1"] == r.var_mu1
        assert row["var_mu2"] == r.var_mu2
        assert row["robust_quantile"] == r.robust_quantile
        assert row["gaussian_quantile"] == r.gaussian_quantile


def test_het_final_periods_sorted(het_result):
    assert het_result.summary["final_period"].to_list() == [2, 3]


def test_het_att_equals_mu1_minus_mu2(het_result):
    for row in het_result.summary.iter_rows(named=True):
        assert row["att"] == pytest.approx(row["mu1"] - row["mu2"], abs=1e-10)


def test_het_var_att_equals_var_sum(het_result):
    for row in het_result.summary.iter_rows(named=True):
        assert row["var_att"] == pytest.approx(row["var_mu1"] + row["var_mu2"], abs=1e-10)


@pytest.mark.filterwarnings("ignore:ds1 contains one element:UserWarning")
def test_het_matches_single_call(estimator_panel):
    het = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1],
        ds2=[0],
        final_periods=[3],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    single = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1],
        ds2=[0],
        final_period=3,
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert het.results[0].att == pytest.approx(single.att, abs=1e-10)


@pytest.mark.parametrize(
    "final_periods, match",
    [
        ([], "non-empty"),
    ],
)
def test_het_invalid_periods_raise(estimator_panel, final_periods, match):
    with pytest.raises(ValueError, match=match):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1],
            ds2=[0],
            final_periods=final_periods,
            xformla="~ X1",
        )


def test_het_repr_contains_table(het_result):
    text = str(het_result)
    assert "ATE" in text
    assert "Period" in text


@pytest.mark.parametrize("h", [2, 3])
def test_impulse_response_matches_manual_ds(impulse_panel, h):
    ir = dyn_balancing(
        data=impulse_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1, 1],
        ds2=[0, 0, 0],
        histories_length=[h],
        impulse_response=True,
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    expected_ds1 = [1] + [0] * (h - 1)
    expected_ds2 = [0] * h
    manual = dyn_balancing(
        data=impulse_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=expected_ds1,
        ds2=expected_ds2,
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert ir.results[0].att == pytest.approx(manual.att, abs=1e-10)
    assert ir.results[0].mu1 == pytest.approx(manual.mu1, abs=1e-10)
    assert ir.results[0].mu2 == pytest.approx(manual.mu2, abs=1e-10)


def test_impulse_response_att_equals_mu_diff(impulse_result):
    for row in impulse_result.summary.iter_rows(named=True):
        assert row["att"] == pytest.approx(row["mu1"] - row["mu2"], abs=1e-10)


def test_impulse_response_var_att_equals_var_sum(impulse_result):
    for row in impulse_result.summary.iter_rows(named=True):
        assert row["var_att"] == pytest.approx(row["var_mu1"] + row["var_mu2"], abs=1e-10)


def test_impulse_response_ignores_original_ds(impulse_panel):
    r1 = dyn_balancing(
        data=impulse_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1, 1],
        ds2=[0, 0, 0],
        histories_length=[2],
        impulse_response=True,
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    r2 = dyn_balancing(
        data=impulse_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 0, 0],
        ds2=[1, 1, 1],
        histories_length=[2],
        impulse_response=True,
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert r1.results[0].att == pytest.approx(r2.results[0].att, abs=1e-10)


def test_impulse_response_without_histories_raises(estimator_panel):
    with pytest.raises(ValueError, match="requires histories_length"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            impulse_response=True,
            xformla="~ X1",
        )


def test_pooled_counts_original_units(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ X1",
        pooled=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.estimation_params["n_units"] == 60
    assert result.estimation_params["n_stacked_units"] == 120
    assert "Stacked unit histories: 120" in str(result)


def test_pooled_initial_period_before_first_window(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ X1",
        pooled=True,
        initial_period=1,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert np.isfinite(result.att)
    assert result.estimation_params["n_stacked_units"] == 120


def test_unpooled_initial_period_is_ignored(estimator_panel):
    kwargs = dict(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    with pytest.warns(UserWarning, match="initial_period only applies"):
        result = dyn_balancing(initial_period=1, **kwargs)
    assert result.att == dyn_balancing(**kwargs).att


def test_outcome_missing_before_window_leaves_estimate_unchanged(estimator_panel):
    kwargs = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ X1",
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    missing = (pl.col("id") < 5) & (pl.col("time") == 1)
    nulled = estimator_panel.with_columns(pl.when(missing).then(None).otherwise(pl.col("y")).alias("y"))
    base = dyn_balancing(data=estimator_panel, **kwargs)
    result = dyn_balancing(data=nulled, **kwargs)
    assert result.att == pytest.approx(base.att, abs=1e-12)
    assert result.estimation_params["n_units"] == 60


def test_negative_lags_raise(estimator_panel):
    with pytest.raises(ValueError, match="lags must be a nonnegative integer"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla="~ X1",
            lags=-1,
        )


def test_ipw_msm_raises(estimator_panel):
    with pytest.raises(ValueError, match="balancing='ipw_msm' is not available"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla="~ X1",
            balancing="ipw_msm",
        )


def test_demeaned_fe_raises(estimator_panel):
    with pytest.raises(NotImplementedError, match="demeaned_fe=True applies only to continuous treatments"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla="~ X1",
            demeaned_fe=True,
        )


def test_several_clustervars_raise(estimator_panel):
    with pytest.raises(ValueError, match="only one-way clustering is supported"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla="~ X1",
            clustervars=["cluster_var", "id"],
        )


def test_clustervars_string_matches_list(estimator_panel):
    kwargs = dict(
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
    assert (
        dyn_balancing(clustervars="cluster_var", **kwargs).se == dyn_balancing(clustervars=["cluster_var"], **kwargs).se
    )


@pytest.mark.parametrize("xformla", [None, "~1"])
def test_no_covariates_raise(estimator_panel, xformla):
    with pytest.raises(ValueError, match="needs at least one"):
        dyn_balancing(
            data=estimator_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[0, 1, 1],
            ds2=[0, 0, 0],
            xformla=xformla,
        )


def test_estimation_params_store_alpha_and_robust_quantile(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        alp=0.1,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.estimation_params["alpha"] == 0.1
    assert result.estimation_params["robust_quantile"] is False
    assert "alp" not in result.estimation_params
    assert "90% Conf. Interval" in str(result)


def test_default_critical_value_is_gaussian(estimator_panel):
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
    assert result.robust_quantile == result.gaussian_quantile
    assert "Gaussian critical values" in str(result)


def test_robust_quantile_prints_chi_squared_interval(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        robust_quantile=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    assert result.robust_quantile == pytest.approx(np.sqrt(chi2.ppf(0.95, 6)))
    assert f"{result.att - result.robust_quantile * result.se:.4f}" in str(result)
    assert "Robust (chi-squared) critical values" in str(result)


def test_imbalances_cover_every_period_and_covariate(estimator_panel):
    panel = estimator_panel.with_columns((pl.col("id") % 3).alias("fe_group"))
    result = dyn_balancing(
        data=panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1 + X2",
        fixed_effects=["fe_group"],
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    table = result.imbalances["ds2"]
    assert set(result.imbalances) == {"ds1", "ds2"}
    assert table.columns == ["period", "covariate", "imbalance"]
    assert table.height == 15
    assert table.filter(pl.col("period") == 1)["covariate"].to_list() == [
        "X1",
        "X2",
        "fe_group_0",
        "fe_group_1",
        "fe_group_2",
    ]


def test_pooled_imbalances_name_time_dummies_after_time_column(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ X1",
        fixed_effects=["time"],
        pooled=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    labels = result.imbalances["ds1"].filter(pl.col("period") == 1)["covariate"].to_list()
    assert labels == ["X1", "time_1", "time_2", "time_3"]


def test_imbalances_match_weights(estimator_panel):
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
    gammas = result.gammas["ds1"]
    for t in range(3):
        x = estimator_panel.filter(pl.col("time") == t + 1).sort("id").select("X1", "X2").to_numpy()
        previous = np.full(60, 1 / 60) if t == 0 else gammas[:, t - 1]
        expected = (gammas[:, t] - previous) @ x / x.std(axis=0, ddof=1)
        stored = result.imbalances["ds1"].filter(pl.col("period") == t + 1)["imbalance"].to_numpy()
        np.testing.assert_allclose(stored, expected, rtol=1e-12, atol=1e-15)


def test_ipw_standard_errors_follow_clustervars(estimator_panel):
    kwargs = dict(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        balancing="ipw",
    )
    plain = dyn_balancing(**kwargs)
    clustered = dyn_balancing(clustervars=["cluster_var"], **kwargs)
    assert clustered.att == plain.att
    assert clustered.se != pytest.approx(plain.se)


def test_debias_is_reproducible_with_random_state(estimator_panel):
    kwargs = dict(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        debias=True,
        regularization=False,
        ub=20.0,
        grid_length=50,
        adaptive_balancing=False,
    )
    first = dyn_balancing(random_state=7, **kwargs)
    second = dyn_balancing(random_state=7, **kwargs)
    other = dyn_balancing(random_state=8, **kwargs)
    assert first.att == second.att
    assert first.att != other.att


def test_converter_reads_alpha_of_estimate(estimator_panel):
    result = dyn_balancing(
        data=estimator_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1, 1],
        ds2=[0, 0, 0],
        xformla="~ X1",
        alp=0.1,
        robust_quantile=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    row = dynbalancingresult_to_polars(result).filter(pl.col("parameter") == "mu(ds1)").row(0, named=True)
    assert (row["ci_upper_robust"] - row["estimate"]) / row["se"] == pytest.approx(np.sqrt(chi2.ppf(0.9, 3)))


def test_het_converter_gives_potential_outcomes_their_own_degrees_of_freedom(impulse_panel):
    result = dyn_balancing(
        data=impulse_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 0],
        ds2=[0, 0],
        final_periods=[2, 3],
        xformla="~ X1",
        alp=0.1,
        robust_quantile=True,
        ub=20.0,
        grid_length=50,
        nfolds=3,
        adaptive_balancing=False,
    )
    mu = dynbalancinghetresult_to_polars(result, parameter="mu2")
    ate = dynbalancinghetresult_to_polars(result)
    np.testing.assert_allclose(
        ((mu["ci_upper_robust"] - mu["estimate"]) / mu["se"]).to_numpy(), np.sqrt(chi2.ppf(0.9, 2))
    )
    np.testing.assert_allclose(
        ((ate["ci_upper_robust"] - ate["estimate"]) / ate["se"]).to_numpy(), np.sqrt(chi2.ppf(0.9, 4))
    )
