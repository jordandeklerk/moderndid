# pylint: disable=no-self-use
"""Tests for DiD preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from pydid.did import preprocess_did
from pydid.did.preprocess.constants import (
    NEVER_TREATED_VALUE,
    ControlGroup,
)
from pydid.did.preprocess.models import DIDConfig, DIDData
from pydid.did.preprocess.tensors import TensorFactorySelector
from pydid.did.preprocess.transformers import (
    DataTransformerPipeline,
    TreatmentEncoder,
    WeightNormalizer,
)
from pydid.did.preprocess.validators import (
    ArgumentValidator,
    ColumnValidator,
    CompositeValidator,
    TreatmentValidator,
)


def create_test_panel_data(
    n_units=100,
    n_periods=4,
    treat_fraction=0.5,
    treat_period=3,
    seed=42,
):
    np.random.seed(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    n_treated = int(n_units * treat_fraction)
    treated_units = np.random.choice(n_units, n_treated, replace=False)

    treated = np.isin(units, treated_units).astype(int)
    post = (periods >= treat_period).astype(int)
    d = treated * post

    g = np.zeros(len(units))
    for unit in treated_units:
        g[units == unit] = treat_period

    unit_fe = np.random.normal(0, 1, n_units)[units]
    time_fe = np.random.normal(0, 0.5, n_periods)[periods - 1]
    treatment_effect = 2.0
    y = unit_fe + time_fe + d * treatment_effect + np.random.normal(0, 0.5, len(units))

    x1 = np.random.normal(0, 1, len(units))
    x2 = np.random.normal(0, 1, len(units))

    df = pd.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "g": g,
            "x1": x1,
            "x2": x2,
        }
    )

    return df


def create_test_repeated_cross_section(
    n_per_period=100,
    n_periods=4,
    treat_fraction=0.5,
    treat_period=3,
    seed=42,
):
    np.random.seed(seed)

    n_total = n_per_period * n_periods

    periods = np.repeat(np.arange(1, n_periods + 1), n_per_period)

    treated = np.random.binomial(1, treat_fraction, n_total)
    post = (periods >= treat_period).astype(int)
    d = treated * post

    g = np.where(treated == 1, treat_period, 0)

    time_fe = np.random.normal(0, 0.5, n_periods)[periods - 1]
    treatment_effect = 2.0
    y = time_fe + treated * 0.5 + d * treatment_effect + np.random.normal(0, 1, n_total)

    x1 = np.random.normal(0, 1, n_total)
    x2 = np.random.normal(0, 1, n_total)

    df = pd.DataFrame(
        {
            "time": periods,
            "y": y,
            "g": g,
            "x1": x1,
            "x2": x2,
        }
    )

    return df


class TestValidators:
    def test_column_validator(self):
        df = create_test_panel_data()
        validator = ColumnValidator()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
        )
        result = validator.validate(df, config)
        assert result.is_valid

        config = DIDConfig(
            yname="missing_column",
            tname="time",
            idname="id",
            gname="g",
        )
        result = validator.validate(df, config)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_treatment_validator(self):
        df = create_test_panel_data()
        validator = TreatmentValidator()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
        )

        result = validator.validate(df, config)
        assert result.is_valid

        df_invalid = df.copy()
        df_invalid.loc[(df_invalid["id"] == 0) & (df_invalid["time"] == 4), "g"] = 0
        result = validator.validate(df_invalid, config)
        assert not result.is_valid

    def test_argument_validator(self):
        df = create_test_panel_data()
        validator = ArgumentValidator()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            control_group=ControlGroup.NEVER_TREATED,
            base_period="universal",
            anticipation=0,
            alp=0.05,
        )
        result = validator.validate(df, config)
        assert result.is_valid

        config_invalid_alpha = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            alp=1.5,
        )
        result = validator.validate(df, config_invalid_alpha)
        assert not result.is_valid
        assert any("alp must be between 0 and 1" in err for err in result.errors)

        config_negative_anticipation = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            anticipation=-1,
        )
        result = validator.validate(df, config_negative_anticipation)
        assert not result.is_valid
        assert any("anticipation must be positive" in err for err in result.errors)

    def test_composite_validator(self):
        df = create_test_panel_data()
        validator = CompositeValidator()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            control_group=ControlGroup.NEVER_TREATED,
        )

        result = validator.validate(df, config)
        assert result.is_valid


class TestTransformers:
    def test_weight_normalizer(self):
        df = create_test_panel_data()
        transformer = WeightNormalizer()

        config = DIDConfig(yname="y", tname="time", gname="g")

        df_transformed = transformer.transform(df, config)
        assert "weights" in df_transformed.columns
        assert np.isclose(df_transformed["weights"].mean(), 1.0)

    def test_treatment_encoder(self):
        df = create_test_panel_data()
        transformer = TreatmentEncoder()

        config = DIDConfig(yname="y", tname="time", gname="g")

        df_transformed = transformer.transform(df, config)
        assert np.any(np.isinf(df_transformed["g"]))
        assert np.all(df_transformed.loc[df_transformed["g"] != np.inf, "g"] > 0)

    def test_transformer_pipeline(self):
        df = create_test_panel_data()
        pipeline = DataTransformerPipeline()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
            allow_unbalanced_panel=True,
        )

        df_transformed = pipeline.transform(df, config)

        assert "weights" in df_transformed.columns
        assert np.any(np.isinf(df_transformed["g"]))
        assert len(df_transformed) > 0

        assert config.time_periods_count > 0
        assert config.treated_groups_count > 0
        assert config.id_count > 0


class TestTensorFactories:
    def test_panel_tensor_factory(self):
        df = create_test_panel_data()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
            allow_unbalanced_panel=False,
        )

        pipeline = DataTransformerPipeline()
        df_transformed = pipeline.transform(df, config)

        tensors = TensorFactorySelector.create_tensors(df_transformed, config)

        assert tensors["outcomes_tensor"] is not None
        assert len(tensors["outcomes_tensor"]) == config.time_periods_count
        assert all(len(y) == config.id_count for y in tensors["outcomes_tensor"])

        assert tensors["time_invariant_data"] is not None
        assert len(tensors["time_invariant_data"]) == config.id_count
        assert tensors["weights"] is not None
        assert len(tensors["weights"]) == config.id_count

    def test_rcs_tensor_factory(self):
        df = create_test_repeated_cross_section()

        config = DIDConfig(
            yname="y",
            tname="time",
            gname="g",
            panel=False,
            xformla="~ x1 + x2",
        )

        pipeline = DataTransformerPipeline()
        df_transformed = pipeline.transform(df, config)

        tensors = TensorFactorySelector.create_tensors(df_transformed, config)

        assert tensors["outcomes_tensor"] is None

        assert tensors["covariates_matrix"] is not None
        assert tensors["covariates_matrix"].shape[1] == 3


class TestPreprocessDid:
    def test_basic_preprocessing(self):
        df = create_test_panel_data()

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
            allow_unbalanced_panel=True,
            print_details=False,
        )

        assert isinstance(result, DIDData)

        assert result.data is not None
        assert result.weights is not None
        assert result.config.yname == "y"
        assert result.config.panel is True

        assert result.config.time_periods_count > 0
        assert result.config.treated_groups_count > 0
        assert result.config.id_count > 0

    def test_with_all_options(self):
        df = create_test_panel_data()

        df["w"] = np.random.uniform(0.5, 1.5, len(df))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ x1 + x2",
            panel=True,
            allow_unbalanced_panel=False,
            control_group="notyettreated",
            anticipation=1,
            weightsname="w",
            clustervars=["id"],
            est_method="ipw",
            base_period="universal",
            print_details=False,
        )

        assert result.config.control_group == ControlGroup.NOT_YET_TREATED
        assert result.config.anticipation == 1
        assert result.config.weightsname == "w"
        assert result.config.est_method.value == "ipw"
        assert result.cluster is not None
        assert np.array_equal(result.cluster, result.time_invariant_data["id"].values)
        assert result.config.clustervars == ["id"]

        assert result.outcomes_tensor is not None
        assert result.covariates_tensor is not None

    def test_repeated_cross_section_preprocessing(self):
        df = create_test_repeated_cross_section()

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            gname="g",
            panel=False,
            print_details=False,
        )

        assert result.config.true_repeated_cross_sections is True
        assert result.config.idname == ".rowid"
        assert result.outcomes_tensor is None
        assert result.covariates_matrix is not None

    def test_no_never_treated(self):
        df = create_test_panel_data(n_periods=6, treat_period=3)
        df.loc[(df["id"].isin(range(70, 80))) & (df["g"] == 0), "g"] = 5
        df = df[df["g"] != 0].copy()

        with pytest.warns(UserWarning, match="No never-treated group"):
            result = preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                control_group="nevertreated",
                print_details=False,
            )

        assert np.any(result.data["g"] == NEVER_TREATED_VALUE)
        assert len(result.config.treated_groups) > 0

    def test_empty_groups_error(self):
        df = create_test_panel_data()
        df["g"] = 1

        with pytest.raises(ValueError, match="No valid time periods remaining|No valid groups"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                print_details=False,
            )

    def test_invalid_control_group(self):
        df = create_test_panel_data()

        with pytest.raises(ValueError, match="'invalid' is not a valid ControlGroup"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                control_group="invalid",
                print_details=False,
            )
