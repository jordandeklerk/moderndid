"""Tests for dynamic covariate balancing preprocessing."""

import warnings

import numpy as np
import polars as pl
import pytest

from moderndid.core.preprocess import DynBalancingConfig, DynBalancingData
from tests.diddynamic.conftest import build_dyn_balancing


@pytest.fixture
def base_config():
    return dict(yname="y", tname="time", idname="id", treatment_name="D", ds1=[3], ds2=[4])


def test_config_default_values():
    cfg = DynBalancingConfig()
    assert cfg.balancing == "dcb"
    assert cfg.method == "lasso_plain"
    assert cfg.adaptive_balancing is True
    assert cfg.nfolds == 10
    assert cfg.grid_length == 1000


def test_config_custom_values():
    cfg = DynBalancingConfig(yname="outcome", tname="period", idname="unit", treatment_name="treat", ds1=[3], ds2=[4])
    assert cfg.yname == "outcome"
    assert cfg.ds1 == [3]
    assert cfg.ds2 == [4]


def test_config_to_dict():
    cfg = DynBalancingConfig(yname="y")
    d = cfg.to_dict()
    assert d["yname"] == "y"
    assert isinstance(d, dict)


def test_builder_returns_dyn_balancing_data(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert isinstance(result, DynBalancingData)


def test_builder_populates_config(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.n_units == 10
    assert result.config.n_periods == 4


def test_treatment_matrix_shape(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix.shape == (10, 4)


def test_treatment_matrix_values(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.treatment_matrix[0, 0] == 0.0
    assert result.treatment_matrix[0, 2] == 1.0
    assert result.treatment_matrix[5, 2] == 0.0


def test_outcome_vector_length(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert len(result.outcome_vector) == 10


def test_outcome_vector_values(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    expected = simple_panel.filter(pl.col("time") == 4).sort("id")["y"].to_numpy()
    np.testing.assert_array_almost_equal(result.outcome_vector, expected)


def test_with_covariates(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert result.has_covariates
    assert result.config.covariate_names == ["X1", "X2"]


def test_covariate_dict_shape(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    assert len(result.covariate_dict) == 4
    for mat in result.covariate_dict.values():
        assert mat.shape == (10, 2)


def test_without_covariates(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert not result.has_covariates


def test_with_cluster(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, clustervars=["cluster_var"])
    assert result.has_cluster
    assert len(result.cluster) == 10


def test_cluster_same_values_same_index(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, clustervars=["cluster_var"])
    assert result.cluster[0] == result.cluster[1]


def test_without_cluster(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert not result.has_cluster


def test_with_fixed_effects(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"])
    assert result.dim_fe > 0


def test_fe_dummies_in_covariate_dict(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, fixed_effects=["cluster_var"])
    final_period = result.config.final_period
    final_mat = result.covariate_dict[final_period]
    assert final_mat.shape[1] > 0


def test_drops_incomplete_units(unbalanced_panel):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = build_dyn_balancing(
            unbalanced_panel, yname="y", tname="time", idname="id", treatment_name="D", ds1=[2], ds2=[3]
        )
    assert 1 not in result.panel["id"].to_list()
    assert result.n_units == 2


def test_balanced_panel_unchanged(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.n_units == 10


def test_final_period_auto(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.final_period == 4


def test_initial_period_auto(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert result.config.initial_period == 1


def test_panel_stored(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config)
    assert isinstance(result.panel, pl.DataFrame)
    assert result.panel.shape[0] > 0


@pytest.mark.parametrize("col", ["yname", "treatment_name"])
def test_missing_required_column_raises(simple_panel, col):
    config = dict(yname="y", tname="time", idname="id", treatment_name="D", ds1=[3], ds2=[4])
    config[col] = "nonexistent"
    with pytest.raises(ValueError, match="not found in data"):
        build_dyn_balancing(simple_panel, **config)


def test_missing_covariate_raises(simple_panel, base_config):
    with pytest.raises(ValueError, match="not in the dataset"):
        build_dyn_balancing(simple_panel, **base_config, xformla="~nonexistent")


@pytest.mark.parametrize("col", ["idname", "tname"])
def test_missing_id_or_time_column_raises(simple_panel, col):
    config = dict(yname="y", tname="time", idname="id", treatment_name="D", ds1=[3], ds2=[4])
    config[col] = "nonexistent"
    with pytest.raises(ValueError, match="not found in data"):
        build_dyn_balancing(simple_panel, **config)


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


def test_covariate_matrices_contain_no_nan(simple_panel, base_config):
    result = build_dyn_balancing(simple_panel, **base_config, xformla="~X1+X2")
    for mat in result.covariate_dict.values():
        assert not np.any(np.isnan(mat))


def test_covariates_with_fixed_effects_and_cluster(simple_panel):
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
    assert result.dim_fe > 0


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
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
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
