"""Tests for dynamic covariate balancing preprocessing."""

import re

import numpy as np
import polars as pl
import pytest

from moderndid.core.preprocess import DynBalancingConfig, DynBalancingData
from moderndid.core.preprocess.transformers import MissingDataHandler
from tests.diddynamic.conftest import build_dyn_balancing, dropped_messages


def test_default_balancing():
    cfg = DynBalancingConfig()
    assert cfg.balancing == "dcb"


def test_default_method():
    cfg = DynBalancingConfig()
    assert cfg.method == "lasso_plain"


def test_default_adaptive_balancing():
    cfg = DynBalancingConfig()
    assert cfg.adaptive_balancing is True


def test_default_nfolds():
    cfg = DynBalancingConfig()
    assert cfg.nfolds == 10


def test_default_grid_length():
    cfg = DynBalancingConfig()
    assert cfg.grid_length == 1000


def test_default_regularization():
    cfg = DynBalancingConfig()
    assert cfg.regularization is True


def test_default_debias():
    cfg = DynBalancingConfig()
    assert cfg.debias is False


def test_default_robust_quantile():
    cfg = DynBalancingConfig()
    assert cfg.robust_quantile is False


def test_custom_names():
    cfg = DynBalancingConfig(
        yname="outcome",
        tname="period",
        idname="unit",
        treatment_name="treat",
        ds1=[0, 0, 1, 1],
        ds2=[0, 0, 0, 0],
    )
    assert cfg.yname == "outcome"
    assert cfg.tname == "period"
    assert cfg.idname == "unit"
    assert cfg.treatment_name == "treat"


def test_custom_ds():
    cfg = DynBalancingConfig(ds1=[0, 0, 1, 1], ds2=[0, 0, 0, 0])
    assert cfg.ds1 == [0, 0, 1, 1]
    assert cfg.ds2 == [0, 0, 0, 0]


def test_to_dict_returns_dict():
    cfg = DynBalancingConfig(yname="y")
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["yname"] == "y"


def test_to_dict_contains_all_fields():
    cfg = DynBalancingConfig(yname="y", balancing="ipw")
    d = cfg.to_dict()
    assert d["balancing"] == "ipw"
    assert "nfolds" in d


def test_returns_dyn_balancing_data(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert isinstance(result, DynBalancingData)


def test_populates_n_units(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.n_units == 10


def test_populates_n_periods(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.n_periods == 4


def test_panel_stored_as_polars(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert isinstance(result.panel, pl.DataFrame)
    assert result.panel.shape[0] > 0


def test_shape(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix.shape == (10, 4)


def test_treated_unit_period3(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix[0, 2] == 1.0


def test_untreated_unit_period1(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix[0, 0] == 0.0


def test_control_unit_stays_zero(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix[5, 2] == 0.0


def test_length_matches_units(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert len(result.outcome_vector) == 10


def test_values_match_final_period(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    expected = simple_panel.filter(pl.col("time") == 4).sort("id")["y"].to_numpy()
    np.testing.assert_array_almost_equal(result.outcome_vector, expected)


def test_with_covariates_flag(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert result.has_covariates


def test_covariate_names(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert result.config.covariate_names == ["X1", "X2"]


def test_covariate_dict_length(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert len(result.covariate_dict) == 4


@pytest.mark.parametrize("period", [1, 2, 3, 4])
def test_covariate_dict_shapes(simple_panel, base_config, period):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert result.covariate_dict[period].shape == (10, 2)


def test_without_covariates_flag(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert not result.has_covariates


def test_covariate_matrices_no_nan(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    for mat in result.covariate_dict.values():
        assert not np.any(np.isnan(mat))


def test_dotted_and_backticked_covariate_names(simple_panel, base_config):
    renamed = simple_panel.with_columns(pl.col("X1").alias("lag1.X1"), pl.col("X2").alias("X 2"))
    result = build_dyn_balancing(renamed, **base_config, xformla="~ lag1.X1 + `X 2`")
    expected = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")

    assert result.config.covariate_names == ["lag1.X1", "X 2"]
    for period, mat in expected.covariate_dict.items():
        np.testing.assert_array_equal(result.covariate_dict[period], mat)


@pytest.mark.parametrize(
    "xformla, term",
    [("~ X1 + I(X1**2)", "I(X1**2)"), ("~ log(X2)", "log(X2)"), ("~ X1 + X1:X2", "X1:X2")],
)
def test_transformed_covariates_raise(simple_panel, base_config, xformla, term):
    with pytest.raises(ValueError, match=re.escape(f"xformla term '{term}' is not a column name")):
        build_dyn_balancing(simple_panel, **base_config, xformla=xformla)


def test_intercept_only_formula(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~1")
    assert not result.has_covariates


def test_with_cluster_flag(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, clustervars=["cluster_var"])
    assert result.has_cluster


def test_cluster_length(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, clustervars=["cluster_var"])
    assert len(result.cluster) == 10


def test_same_cluster_for_same_group(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, clustervars=["cluster_var"])
    assert result.cluster[0] == result.cluster[1]


def test_without_cluster_flag(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert not result.has_cluster


def test_with_fixed_effects(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"])
    final_mat = result.covariate_dict[result.config.final_period]
    assert final_mat.shape[1] > 0
    assert result.dim_fe == 0


def test_demeaned_fe_sets_dim_fe(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"], demeaned_fe=True)
    assert result.dim_fe > 0


def test_fe_dummies_in_covariate_dict(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"])
    final_mat = result.covariate_dict[result.config.final_period]
    assert final_mat.shape[1] > 0


def test_fe_dummies_in_all_periods(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"])
    widths = {p: mat.shape[1] for p, mat in result.covariate_dict.items()}
    assert len(set(widths.values())) == 1


def test_auto_final_period(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.final_period == 4


def test_auto_initial_period(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.initial_period is None
    assert result.config.time_periods.tolist() == [1, 2, 3, 4]


def test_explicit_final_period(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1, 1],
        ds2=[0, 0, 0],
        final_period=3,
    )
    assert result.config.final_period == 3


def test_explicit_initial_and_final_period(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        final_period=4,
        initial_period=3,
        pooled=True,
    )
    assert result.config.initial_period == 3
    assert result.config.final_period == 4


@pytest.mark.filterwarnings("ignore:Dropped.*units:UserWarning")
def test_drops_incomplete_units(unbalanced_panel):
    result = build_dyn_balancing(
        unbalanced_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[2],
        ds2=[3],
    )
    assert 1 not in result.panel["id"].to_list()
    assert result.n_units == 2


def test_balanced_panel_unchanged(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.n_units == 10


def test_panel_with_nan_in_covariates():
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "D": [0.0, 1.0, 0.0, 0.0],
            "X1": [0.1, None, 0.3, 0.4],
        }
    )
    result = build_dyn_balancing(
        df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1],
        ds2=[2],
        xformla="~X1",
    )
    assert result.n_units >= 1


@pytest.mark.parametrize("col", ["yname", "treatment_name"])
def test_missing_required_column_raises(simple_panel, col):
    config = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 0, 1, 1],
        ds2=[0, 0, 0, 0],
    )
    config[col] = "nonexistent"
    with pytest.raises(ValueError, match=f"^Validation failed:\n{col}='nonexistent' is not a column in the data\\.$"):
        build_dyn_balancing(simple_panel, **config)


@pytest.mark.parametrize("col", ["idname", "tname"])
def test_missing_id_or_time_column_raises(simple_panel, col):
    config = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 0, 1, 1],
        ds2=[0, 0, 0, 0],
    )
    config[col] = "nonexistent"
    with pytest.raises(ValueError, match=f"^Validation failed:\n{col}='nonexistent' is not a column in the data\\.$"):
        build_dyn_balancing(simple_panel, **config)


def test_missing_covariate_raises(simple_panel, base_config):
    with pytest.raises(
        ValueError, match="^Validation failed:\n'nonexistent' in xformla is not a column in the data\\.$"
    ):
        build_dyn_balancing(simple_panel, **base_config, xformla="~nonexistent")


def test_covariates_with_fe_and_cluster(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[3],
        ds2=[4],
        xformla="~X1+X2",
        fixed_effects=["cluster_var"],
        clustervars=["cluster_var"],
    )
    assert result.has_covariates
    assert result.has_cluster
    final_mat = result.covariate_dict[result.config.final_period]
    assert final_mat.shape[1] > 2


def test_constant_outcome_preserved():
    n_units = 6
    n_periods = 2
    ids = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)
    df = pl.DataFrame(
        {
            "id": ids,
            "time": times,
            "y": np.full(n_units * n_periods, 7.5),
            "D": np.zeros(n_units * n_periods),
        }
    )
    result = build_dyn_balancing(
        df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1],
        ds2=[0],
    )
    np.testing.assert_array_almost_equal(result.outcome_vector, np.full(n_units, 7.5))


def test_treatment_assignment_known():
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1, 2, 2],
            "time": [1, 2, 1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "D": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    result = build_dyn_balancing(
        df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[0, 1],
        ds2=[0, 0],
    )
    assert result.treatment_matrix.shape == (3, 2)
    assert result.treatment_matrix[0, 0] == 0.0
    assert result.treatment_matrix[0, 1] == 1.0


def test_non_numeric_treatment_column():
    df = pl.DataFrame(
        {
            "id": [0, 0, 1, 1],
            "time": [1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0],
            "D": [0, 1, 0, 0],
        }
    )
    result = build_dyn_balancing(
        df,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1],
        ds2=[0],
    )
    assert result.treatment_matrix.dtype == np.float64 or np.issubdtype(result.treatment_matrix.dtype, np.number)


def test_pooled_expands_panel(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    assert result.n_units > 10


def test_pooled_false_unchanged(simple_panel, base_config):
    result_no_pool = build_dyn_balancing(simple_panel, **base_config, pooled=False)
    assert result_no_pool.n_units == 10


def test_pooled_has_new_name_column(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    assert "new_name" in result.panel.columns


def test_pooled_has_new_time_column(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    assert "new_Time" in result.panel.columns


def test_pooled_preserves_original_rows(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    original_ids = set(str(i) for i in range(10))
    panel_ids = set(result.panel["new_name"].unique().to_list())
    assert original_ids.issubset(panel_ids)


def test_pooled_pseudo_units_have_complete_history(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    counts = result.panel.group_by("new_name").len()
    assert (counts["len"] >= 2).all()


def test_pooled_time_fe_uses_new_time(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
        fixed_effects=["cluster_var", "time"],
    )
    fe_cols = [c for c in result.panel.columns if c.startswith("new_Time_")]
    assert len(fe_cols) > 0


def test_pooled_without_time_fe_no_rename(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
        fixed_effects=["cluster_var"],
    )
    fe_cols = [c for c in result.panel.columns if c.startswith("new_Time_")]
    assert len(fe_cols) == 0


def test_pooled_treatment_matrix_expanded(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    assert result.treatment_matrix.shape[0] == result.n_units
    assert result.treatment_matrix.shape[1] >= 2


def test_pooled_outcome_vector_length(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
    )
    assert len(result.outcome_vector) == result.n_units


def test_pooled_default_initial_period_is_first_full_window(simple_panel):
    result = build_dyn_balancing(
        simple_panel, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], pooled=True
    )
    assert result.config.initial_period == 2


def test_pooled_default_stacks_every_window(simple_panel):
    result = build_dyn_balancing(
        simple_panel, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], pooled=True
    )
    assert result.n_units == 30


@pytest.mark.parametrize("initial_period, n_units", [(1, 30), (2, 30), (3, 20)])
def test_pooled_initial_period_bounds_windows(simple_panel, initial_period, n_units):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
        initial_period=initial_period,
    )
    assert result.n_units == n_units
    assert result.config.time_periods.tolist() == [3, 4]
    assert result.treatment_matrix.shape == (n_units, 2)


def test_pooled_initial_period_after_final_raises(simple_panel):
    with pytest.raises(ValueError, match="must not be later than final_period"):
        build_dyn_balancing(
            simple_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1, 1],
            ds2=[0, 0],
            pooled=True,
            final_period=3,
            initial_period=4,
        )


def test_pooled_initial_period_not_in_data_raises(simple_panel):
    with pytest.raises(ValueError, match="initial_period=9 is not in the data"):
        build_dyn_balancing(
            simple_panel,
            yname="y",
            tname="time",
            idname="id",
            treatment_name="D",
            ds1=[1, 1],
            ds2=[0, 0],
            pooled=True,
            initial_period=9,
        )


def test_initial_period_without_pooling_warns(simple_panel, base_config):
    with pytest.warns(UserWarning, match="initial_period only applies when pooled=True"):
        result = build_dyn_balancing(simple_panel, **base_config, initial_period=1)
    assert result.config.time_periods.tolist() == [1, 2, 3, 4]


def test_non_consecutive_periods_raise(simple_panel):
    df = simple_panel.with_columns((pl.col("time") * 5 + 1995).alias("time"))
    with pytest.raises(ValueError, match="consecutive integers"):
        build_dyn_balancing(df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0])


@pytest.mark.parametrize("col, time", [("y", 1), ("y", 3), ("D", 1), ("X1", 3)])
def test_missing_value_outside_final_outcome_keeps_unit(simple_panel, col, time):
    missing = (pl.col("id") == 0) & (pl.col("time") == time)
    df = simple_panel.with_columns(pl.when(missing).then(None).otherwise(pl.col(col)).alias(col))
    result = build_dyn_balancing(
        df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], xformla="~X1"
    )
    assert result.n_units == 10


def test_missing_row_outside_window_keeps_unit(simple_panel):
    df = simple_panel.filter(~((pl.col("id") == 0) & (pl.col("time") == 1)))
    result = build_dyn_balancing(df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0])
    assert result.n_units == 10


def test_missing_covariate_in_window_becomes_nan(simple_panel):
    missing = (pl.col("id") == 0) & (pl.col("time") == 3)
    df = simple_panel.with_columns(pl.when(missing).then(None).otherwise(pl.col("X1")).alias("X1"))
    result = build_dyn_balancing(
        df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], xformla="~X1+X2"
    )
    assert np.isnan(result.covariate_dict[3][0, 0])
    assert not np.isnan(result.covariate_dict[3][1:]).any()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_fixed_effect_in_window_leaves_its_dummies_missing(simple_panel, value):
    missing = (pl.col("id") == 0) & (pl.col("time") == 3)
    level = (pl.col("id") % 3).cast(pl.Float64)
    df = simple_panel.with_columns(pl.when(missing).then(value).otherwise(level).alias("fe"))
    result = build_dyn_balancing(
        df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], fixed_effects=["fe"]
    )
    assert [c for c in result.panel.columns if c.startswith("fe_")] == ["fe_0.0", "fe_1.0", "fe_2.0"]
    assert np.isnan(result.covariate_dict[3][0]).all()
    assert not np.isnan(result.covariate_dict[3][1:]).any()
    assert not np.isnan(result.covariate_dict[4]).any()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_missing_final_cluster_drops_the_unit(simple_panel, value):
    missing = (pl.col("id") == 0) & (pl.col("time") == 4)
    cluster = (pl.col("id") % 2).cast(pl.Float64)
    df = simple_panel.with_columns(pl.when(missing).then(value).otherwise(cluster).alias("cl"))
    with pytest.warns(UserWarning) as record:
        result = build_dyn_balancing(
            df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], clustervars=["cl"]
        )
    assert "Dropped 1 rows from original data due to missing values" in dropped_messages(record)
    assert result.n_units == 9
    assert len(np.unique(result.cluster)) == 2


def test_missing_final_outcome_drops_unit(simple_panel):
    missing = (pl.col("id") == 0) & (pl.col("time") == 4)
    df = simple_panel.with_columns(pl.when(missing).then(None).otherwise(pl.col("y")).alias("y"))
    with pytest.warns(UserWarning, match="Dropped 1 units with a missing final-period outcome"):
        result = build_dyn_balancing(
            df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0]
        )
    assert result.n_units == 9
    assert not np.isnan(result.outcome_vector).any()


@pytest.mark.parametrize("value", [None, float("nan")])
def test_missing_treatment_in_window_drops_unit(simple_panel, value):
    missing = (pl.col("id") == 0) & (pl.col("time") == 3)
    df = simple_panel.with_columns(pl.when(missing).then(value).otherwise(pl.col("D")).alias("D"))
    with pytest.warns(UserWarning, match="Dropped 1 units that are not observed in every period"):
        result = build_dyn_balancing(
            df, yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0]
        )
    assert result.n_units == 9


def test_time_fixed_effects_cover_window_periods_only(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        fixed_effects=["time"],
    )
    fe_cols = [c for c in result.panel.columns if c.startswith("time_")]
    assert fe_cols == ["time_3", "time_4"]


def test_pooled_fixed_effect_dummies_have_no_empty_columns(simple_panel):
    result = build_dyn_balancing(
        simple_panel,
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        pooled=True,
        initial_period=3,
        fixed_effects=["cluster_var", "time"],
    )
    fe_cols = [c for c in result.panel.columns if c.startswith("new_Time_")]
    assert fe_cols == ["new_Time_2", "new_Time_3", "new_Time_4"]
    assert all(result.panel[c].sum() > 0 for c in fe_cols)


@pytest.mark.filterwarnings("ignore:Dropped.*units:UserWarning")
@pytest.mark.parametrize("missing", [None, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize(
    ("column", "repeated"),
    [
        pytest.param("time", (pl.col("id") == 1.0) & pl.col("time").is_in([2.0, 3.0]), id="period"),
        pytest.param("id", pl.col("id").is_in([1.0, 2.0]) & (pl.col("time") == 3.0), id="unit"),
    ],
)
def test_rows_missing_the_unit_or_period_leave_and_never_repeat(simple_panel, base_config, column, repeated, missing):
    keyed = simple_panel.with_columns(pl.col("id", "time").cast(pl.Float64))
    with_missing = keyed.with_columns(pl.when(repeated).then(missing).otherwise(pl.col(column)).alias(column))

    with pytest.warns(UserWarning, match="^Dropped 2 rows from original data due to missing values$"):
        result = build_dyn_balancing(with_missing, **base_config)
    expected = build_dyn_balancing(keyed.filter(~repeated), **base_config)

    assert result.panel.equals(expected.panel)
    np.testing.assert_array_equal(result.treatment_matrix, expected.treatment_matrix)
    np.testing.assert_array_equal(result.outcome_vector, expected.outcome_vector)
    assert result.n_units == expected.n_units


def test_unit_that_loses_a_period_to_a_missing_key_leaves_with_the_history_warning(simple_panel, base_config):
    keyed = simple_panel.with_columns(pl.col("id", "time").cast(pl.Float64))
    row = (pl.col("id") == 1.0) & (pl.col("time") == 2.0)
    with_missing = keyed.with_columns(pl.when(row).then(float("nan")).otherwise(pl.col("time")).alias("time"))

    with pytest.warns(UserWarning) as record:
        result = build_dyn_balancing(with_missing, **base_config)

    assert dropped_messages(record) == [
        "Dropped 1 rows from original data due to missing values",
        "Dropped 1 units that are not observed in every period of the treatment history.",
    ]
    assert 1.0 not in result.panel["id"].to_list()
    assert result.n_units == 9


def test_every_row_missing_the_period_raises(simple_panel, base_config):
    data = simple_panel.with_columns(pl.lit(float("nan")).alias("time"))
    message = "Every row has a missing value in 'id' or 'time'. No data is left to estimate from."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        build_dyn_balancing(data, **base_config)


def test_repeated_pair_is_reported_beside_rows_missing_the_unit(simple_panel, base_config):
    keyed = simple_panel.with_columns(pl.col("id", "time").cast(pl.Float64))
    without_unit = keyed.head(2).with_columns(pl.lit(float("inf")).alias("id"))
    repeated = keyed.filter((pl.col("id") == 3.0) & (pl.col("time") == 2.0))

    with pytest.raises(ValueError, match=re.escape("Rows repeat for the (id, time) pair (3.0, 2.0).")):
        build_dyn_balancing(pl.concat([keyed, without_unit, repeated]), **base_config)


@pytest.mark.parametrize(
    "period_dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64],
)
def test_pooled_stacking_does_not_depend_on_the_period_dtype(simple_panel, period_dtype):
    spec = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1, 1],
        ds2=[0, 0, 0],
        pooled=True,
        initial_period=1,
    )
    expected = build_dyn_balancing(simple_panel, **spec)

    result = build_dyn_balancing(simple_panel.with_columns(pl.col("time").cast(period_dtype)), **spec)

    assert result.n_units == expected.n_units == 20
    np.testing.assert_array_equal(result.treatment_matrix, expected.treatment_matrix)
    np.testing.assert_array_equal(result.outcome_vector, expected.outcome_vector)
    np.testing.assert_array_equal(result.config.time_periods, expected.config.time_periods)


@pytest.mark.parametrize("id_dtype", [pl.Int32, pl.UInt16, pl.Float32, pl.Float64])
def test_pooled_stacking_orders_units_the_same_for_every_id_dtype(estimator_panel, id_dtype):
    spec = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~X1",
        pooled=True,
    )
    expected = build_dyn_balancing(estimator_panel, **spec)

    result = build_dyn_balancing(estimator_panel.with_columns(pl.col("id").cast(id_dtype)), **spec)

    assert result.panel["new_name"].unique().sort().to_list() == expected.panel["new_name"].unique().sort().to_list()
    np.testing.assert_array_equal(result.treatment_matrix, expected.treatment_matrix)
    np.testing.assert_array_equal(result.outcome_vector, expected.outcome_vector)
    np.testing.assert_array_equal(
        result.covariate_dict[result.config.final_period], expected.covariate_dict[expected.config.final_period]
    )


def test_pooled_stacking_keeps_fractional_float_ids_apart(estimator_panel):
    spec = dict(
        yname="y",
        tname="time",
        idname="id",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~X1",
        pooled=True,
    )
    expected = build_dyn_balancing(estimator_panel, **spec)

    result = build_dyn_balancing(estimator_panel.with_columns(pl.col("id").cast(pl.Float64) + 0.5), **spec)

    assert result.n_units == expected.n_units == 120
    assert "0.5" in result.panel["new_name"].to_list()


@pytest.mark.filterwarnings("ignore:Dropped.*units:UserWarning")
@pytest.mark.parametrize("infinite", [float("inf"), -float("inf")])
@pytest.mark.parametrize(("column", "time"), [("y", 3), ("y", 4), ("D", 3), ("X1", 3), ("X1", 4)])
def test_infinite_value_in_the_window_builds_the_same_data_as_nan(simple_panel, infinite, column, time):
    spec = dict(yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], xformla="~X1")
    row = (pl.col("id") == 0) & (pl.col("time") == time)
    with_nan = simple_panel.with_columns(pl.when(row).then(float("nan")).otherwise(pl.col(column)).alias(column))
    with_infinity = simple_panel.with_columns(pl.when(row).then(infinite).otherwise(pl.col(column)).alias(column))

    result = build_dyn_balancing(with_infinity, **spec)
    expected = build_dyn_balancing(with_nan, **spec)

    assert result.panel.equals(expected.panel)
    assert result.n_units == expected.n_units
    np.testing.assert_array_equal(result.treatment_matrix, expected.treatment_matrix)
    np.testing.assert_array_equal(result.outcome_vector, expected.outcome_vector)
    assert result.covariate_dict.keys() == expected.covariate_dict.keys()
    for period in expected.covariate_dict:
        np.testing.assert_array_equal(result.covariate_dict[period], expected.covariate_dict[period])


def test_missing_data_step_turns_infinities_into_nan_and_keeps_everything_else():
    config = DynBalancingConfig(
        yname="y", tname="time", idname="id", treatment_name="D", ds1=[1, 1], ds2=[0, 0], xformla="~X1"
    )
    data = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": pl.Series([1.0, float("inf"), None, 2.5], dtype=pl.Float32),
            "D": [0.0, 1.0, float("-inf"), 1.0],
            "X1": [float("nan"), 2.0, 3.0, float("inf")],
            "count": [1, 2, 3, 4],
        }
    )
    expected = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "time": [1, 2, 1, 2],
            "y": pl.Series([1.0, float("nan"), None, 2.5], dtype=pl.Float32),
            "D": [0.0, 1.0, float("nan"), 1.0],
            "X1": [float("nan"), 2.0, 3.0, float("nan")],
            "count": [1, 2, 3, 4],
        }
    )

    result = MissingDataHandler().transform(data, config)

    assert result.schema == data.schema
    assert result.equals(expected)
