"""Tests for dynamic covariate balancing preprocessing."""

import numpy as np
import polars as pl
import pytest

from moderndid.core.preprocess import DynBalancingConfig, DynBalancingData
from tests.diddynamic.conftest import build_dyn_balancing


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
    assert result.config.initial_period == 1


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
    with pytest.raises(ValueError, match="not found in data"):
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
    with pytest.raises(ValueError, match="not found in data"):
        build_dyn_balancing(simple_panel, **config)


def test_missing_covariate_raises(simple_panel, base_config):
    with pytest.raises(ValueError, match="not in the dataset"):
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
