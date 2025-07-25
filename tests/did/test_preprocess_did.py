# pylint: disable=no-self-use
"""Tests for DiD preprocessing functions."""

import numpy as np
import pandas as pd
import pytest

from didpy.did import preprocess_did
from didpy.did.preprocess.builders import DIDDataBuilder
from didpy.did.preprocess.constants import (
    NEVER_TREATED_VALUE,
    BasePeriod,
    ControlGroup,
    EstimationMethod,
)
from didpy.did.preprocess.models import DIDConfig, DIDData
from didpy.did.preprocess.tensors import TensorFactorySelector
from didpy.did.preprocess.transformers import (
    DataTransformerPipeline,
    TreatmentEncoder,
    WeightNormalizer,
)
from didpy.did.preprocess.validators import (
    ArgumentValidator,
    ColumnValidator,
    CompositeValidator,
    PanelStructureValidator,
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


def create_unbalanced_panel_data(n_units=100, n_periods=4, missing_fraction=0.2, seed=42):
    np.random.seed(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    periods = np.tile(np.arange(1, n_periods + 1), n_units)

    n_obs = len(units)
    keep_mask = np.random.uniform(size=n_obs) > missing_fraction

    units = units[keep_mask]
    periods = periods[keep_mask]

    treated_units = np.random.choice(n_units, n_units // 2, replace=False)
    g = np.zeros(len(units))
    for i, unit in enumerate(units):
        if unit in treated_units:
            g[i] = 3

    y = np.random.normal(0, 1, len(units))

    return pd.DataFrame(
        {
            "id": units,
            "time": periods,
            "y": y,
            "g": g,
            "x1": np.random.normal(0, 1, len(units)),
        }
    )


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


class TestEdgeCases:
    def test_missing_data_handling(self):
        df = create_test_panel_data()

        df_missing_y = df.copy()
        df_missing_y.loc[df_missing_y.index[:10], "y"] = np.nan

        result = preprocess_did(
            data=df_missing_y,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            print_details=False,
        )

        assert len(result.data) < len(df)
        assert not result.data["y"].isna().any()

    def test_string_time_periods(self):
        df = create_test_panel_data()
        df["time_str"] = df["time"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"})
        df["g_str"] = df["g"].map({0: "never", 3: "Q3"})

        time_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        g_map = {"never": 0, "Q3": 3}

        df["time_numeric"] = df["time_str"].map(time_map)
        df["g_numeric"] = df["g_str"].map(g_map)

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time_numeric",
            idname="id",
            gname="g_numeric",
            print_details=False,
        )

        assert result.config.time_periods is not None
        assert len(result.config.time_periods) == 4

    def test_single_treated_unit(self):
        df = create_test_panel_data(n_units=100)
        treated_ids = df[df["g"] > 0]["id"].unique()
        keep_ids = np.concatenate([df[df["g"] == 0]["id"].unique(), treated_ids[:1]])
        df = df[df["id"].isin(keep_ids)]

        with pytest.warns(UserWarning, match="Be aware that there are some small groups"):
            result = preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                print_details=False,
            )

        assert result.config.treated_groups_count == 1


class TestDataIntegrity:
    def test_panel_structure_validation(self):
        df = create_test_panel_data()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        validator = PanelStructureValidator()
        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
        )

        result = validator.validate(df, config)
        assert not result.is_valid
        assert any("observed more than once" in err for err in result.errors)

    def test_treatment_reversibility(self):
        df = create_test_panel_data()
        treated_unit = df[df["g"] > 0]["id"].iloc[0]
        df.loc[(df["id"] == treated_unit) & (df["time"] == 4), "g"] = 0

        with pytest.raises(ValueError, match="must be irreversible"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                print_details=False,
            )

    def test_early_treatment_handling(self):
        df = create_test_panel_data(n_periods=5)
        early_treated_units = df["id"].unique()[:10]
        df.loc[df["id"].isin(early_treated_units), "g"] = 0.5

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            print_details=False,
        )

        assert not result.data["id"].isin(early_treated_units).any()


class TestCovariateHandling:
    def test_formula_parsing(self):
        df = create_test_panel_data()
        df["x3"] = df["x1"] * df["x2"]
        df["factor_var"] = np.random.choice(["A", "B", "C"], len(df))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ x1 + x2",
            print_details=False,
        )

        assert result.covariates_tensor is not None
        assert all(cov.shape[1] == 3 for cov in result.covariates_tensor)

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ x1 * x2",
            print_details=False,
        )

        assert result.covariates_tensor is not None
        assert all(cov.shape[1] == 3 for cov in result.covariates_tensor)

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ C(factor_var)",
            print_details=False,
        )

        assert result.covariates_tensor is not None
        assert all(cov.shape[1] == 2 for cov in result.covariates_tensor)

    def test_no_intercept_formula(self):
        df = create_test_panel_data()

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ 0 + x1 + x2",
            print_details=False,
        )

        assert result.covariates_tensor is not None
        assert all(cov.shape[1] == 3 for cov in result.covariates_tensor)


class TestWeightHandling:
    def test_zero_weights(self):
        df = create_test_panel_data()
        df["w"] = np.random.uniform(0.5, 2, len(df))
        df.loc[df.index[:20], "w"] = 0

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            weightsname="w",
            print_details=False,
        )

        assert len(result.data) == len(df)
        assert (result.weights >= 0).all()
        assert (result.weights == 0).sum() == 5

    def test_weight_normalization(self):
        df = create_test_panel_data()
        df["w"] = np.random.uniform(1, 10, len(df))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            weightsname="w",
            print_details=False,
        )

        assert np.isclose(result.weights.mean(), 1.0, rtol=0.02)


class TestUnbalancedPanelHandling:
    def test_unbalanced_to_balanced_conversion(self):
        df = create_unbalanced_panel_data(missing_fraction=0.3)

        result_unbalanced = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            allow_unbalanced_panel=True,
            print_details=False,
        )

        assert not result_unbalanced.is_balanced_panel

        result_balanced = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            allow_unbalanced_panel=False,
            print_details=False,
        )

        assert result_balanced.is_balanced_panel
        assert result_balanced.config.id_count < df["id"].nunique()

    def test_time_invariant_covariate_detection(self):
        df = create_test_panel_data()
        df["time_varying"] = df["time"] * np.random.normal(0, 1, len(df))
        df["time_invariant"] = df.groupby("id")["x1"].transform("first")

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ time_varying + time_invariant",
            allow_unbalanced_panel=True,
            print_details=False,
        )

        if not result.is_balanced_panel:
            assert "time_invariant" in result.time_invariant_data.columns
            assert "time_varying" not in result.time_invariant_data.columns


class TestClusteringOptions:
    def test_multiple_clustering_vars(self):
        df = create_test_panel_data()
        df["cluster1"] = df["id"] // 10
        df["cluster2"] = df["time"] % 2

        with pytest.raises(ValueError, match="You can only provide 1 cluster variable"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                clustervars=["cluster1", "cluster2"],
                print_details=False,
            )

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            clustervars=["cluster1"],
            print_details=False,
        )

        assert result.cluster is not None
        assert result.config.clustervars == ["cluster1"]

    def test_invalid_cluster_var(self):
        df = create_test_panel_data()

        with pytest.raises((ValueError, KeyError), match="not found|Column not found"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                clustervars=["nonexistent_var"],
                print_details=False,
            )


class TestConfigurationOptions:
    def test_anticipation_effects(self):
        df = create_test_panel_data(n_periods=6, treat_period=5)

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            anticipation=1,
            control_group="notyettreated",
            print_details=False,
        )

        assert result.config.anticipation == 1

    def test_base_period_options(self):
        df = create_test_panel_data()

        result_universal = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            base_period="universal",
            print_details=False,
        )

        assert result_universal.config.base_period == BasePeriod.UNIVERSAL

        result_varying = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            base_period="varying",
            print_details=False,
        )

        assert result_varying.config.base_period == BasePeriod.VARYING

    def test_estimation_method_options(self):
        df = create_test_panel_data()

        for method in ["dr", "ipw", "reg"]:
            result = preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                est_method=method,
                print_details=False,
            )

            assert result.config.est_method == EstimationMethod(method)


class TestBuilderPattern:
    def test_builder_validation_errors(self):
        df = create_test_panel_data()
        config = DIDConfig(
            yname="missing_column",
            tname="time",
            idname="id",
            gname="g",
        )

        builder = DIDDataBuilder()
        with pytest.raises(ValueError, match="missing_column"):
            builder.with_data(df).with_config(config).validate()

    def test_builder_chaining(self):
        df = create_test_panel_data()
        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
        )

        builder = DIDDataBuilder()
        result = builder.with_data(df).with_config(config).validate().transform().build()

        assert isinstance(result, DIDData)
        assert result.config == config
