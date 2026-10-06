"""Tests for DiD preprocessing functions."""

import re

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid.core.preprocess import (
    NEVER_TREATED_VALUE,
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    BasePeriod,
    CompositeValidator,
    ContDIDConfig,
    ControlGroup,
    DataTransformerPipeline,
    DIDConfig,
    DIDData,
    EstimationMethod,
    PreprocessDataBuilder,
    TensorFactorySelector,
    preprocess_did,
)
from moderndid.core.preprocess.transformers import (
    TreatmentEncoder,
    WeightNormalizer,
)
from moderndid.core.preprocess.validators import (
    ArgumentValidator,
    ColumnValidator,
    PanelStructureValidator,
    TreatmentValidator,
)
from moderndid.did.aggte import aggte
from moderndid.did.att_gt import att_gt


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

    df = pl.DataFrame(
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

    df = pl.DataFrame(
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

    return pl.DataFrame(
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

        df_invalid = df.clone()
        df_invalid = df_invalid.with_columns(
            pl.when((pl.col("id") == 0) & (pl.col("time") == 4)).then(0).otherwise(pl.col("g")).alias("g")
        )
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
        assert result.is_valid

        config_negative_anticipation = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            anticipation=-1,
        )
        result = validator.validate(df, config_negative_anticipation)
        assert result.is_valid

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
        assert WEIGHTS_COLUMN in df_transformed.columns
        assert np.isclose(df_transformed[WEIGHTS_COLUMN].mean(), 1.0)

    def test_treatment_encoder(self):
        df = create_test_panel_data()
        transformer = TreatmentEncoder()

        config = DIDConfig(yname="y", tname="time", gname="g")

        df_transformed = transformer.transform(df, config)
        assert df_transformed["g"].is_infinite().any()
        assert (df_transformed.filter(pl.col("g").is_finite())["g"] > 0).all()

    def test_transformer_pipeline(self):
        df = create_test_panel_data()
        pipeline = DataTransformerPipeline.get_did_pipeline()

        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=True,
            allow_unbalanced_panel=True,
        )

        df_transformed = pipeline.transform(df, config)

        assert WEIGHTS_COLUMN in df_transformed.columns
        assert df_transformed["g"].is_infinite().any()
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

        pipeline = DataTransformerPipeline.get_did_pipeline()
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

        pipeline = DataTransformerPipeline.get_did_pipeline()
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

        df = df.with_columns(pl.Series("w", np.random.uniform(0.5, 1.5, len(df))))

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
        )

        assert result.config.control_group == ControlGroup.NOT_YET_TREATED
        assert result.config.anticipation == 1
        assert result.config.weightsname == "w"
        assert result.config.est_method.value == "ipw"
        assert result.cluster is not None
        assert np.array_equal(result.cluster, result.time_invariant_data["id"].to_numpy())
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
        )

        assert result.config.true_repeated_cross_sections is True
        assert result.config.idname == ".rowid"
        assert result.outcomes_tensor is None
        assert result.covariates_matrix is not None

    @pytest.mark.filterwarnings("ignore:No never-treated group is available:UserWarning")
    def test_no_never_treated(self):
        df = create_test_panel_data(n_periods=6, treat_period=3)
        df = df.with_columns(
            pl.when(pl.col("id").is_in(list(range(70, 80))) & (pl.col("g") == 0))
            .then(5)
            .otherwise(pl.col("g"))
            .alias("g")
        )
        df = df.filter(pl.col("g") != 0)

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            control_group="nevertreated",
        )

        assert (result.data["g"] == NEVER_TREATED_VALUE).any()
        assert len(result.config.treated_groups) > 0

    @pytest.mark.filterwarnings("ignore:.*units were already treated:UserWarning")
    @pytest.mark.filterwarnings("ignore:Dropped.*units:UserWarning")
    def test_empty_groups_error(self):
        df = create_test_panel_data()
        df = df.with_columns(pl.lit(1).alias("g"))

        with pytest.raises(ValueError, match="No valid time periods remaining|No valid groups"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
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
            )


class TestEdgeCases:
    def test_missing_data_handling(self):
        df = create_test_panel_data()

        df_missing_y = df.clone().with_row_index("_idx")
        df_missing_y = df_missing_y.with_columns(
            pl.when(pl.col("_idx") < 10).then(pl.lit(np.nan)).otherwise(pl.col("y")).alias("y")
        ).drop("_idx")

        result = preprocess_did(
            data=df_missing_y,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
        )

        assert len(result.data) > 0
        assert not result.data["y"].is_null().any()

    def test_string_time_periods(self):
        df = create_test_panel_data()
        df = df.with_columns(
            [
                pl.col("time")
                .cast(pl.Utf8)
                .replace_strict({"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}, default=None)
                .alias("time_str"),
                pl.col("g")
                .cast(pl.Utf8)
                .replace_strict({"0": "never", "0.0": "never", "3": "Q3", "3.0": "Q3"}, default="other")
                .alias("g_str"),
            ]
        )

        df = df.with_columns(
            [
                pl.col("time_str")
                .replace_strict({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}, return_dtype=pl.Int64)
                .alias("time_numeric"),
                pl.col("g_str")
                .replace_strict({"never": 0, "Q3": 3, "other": -1}, return_dtype=pl.Int64)
                .alias("g_numeric"),
            ]
        )

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time_numeric",
            idname="id",
            gname="g_numeric",
        )

        assert result.config.time_periods is not None
        assert len(result.config.time_periods) == 4

    @pytest.mark.filterwarnings("ignore:Be aware that there are some small groups:UserWarning")
    def test_single_treated_unit(self):
        df = create_test_panel_data(n_units=100)
        treated_ids = df.filter(pl.col("g") > 0)["id"].unique().to_numpy()
        control_ids = df.filter(pl.col("g") == 0)["id"].unique().to_numpy()
        keep_ids = np.concatenate([control_ids, treated_ids[:1]])
        df = df.filter(pl.col("id").is_in(keep_ids.tolist()))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
        )

        assert result.config.treated_groups_count == 1


class TestDataIntegrity:
    def test_panel_structure_validation(self):
        df = create_test_panel_data()
        df = pl.concat([df, df.head(1)])

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

    def test_panel_true_with_rcs_data(self):
        df = create_test_repeated_cross_section()
        df = df.with_row_index("id")

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
        assert any("panel=True was specified" in err for err in result.errors)

    def test_panel_false_with_panel_data(self):
        df = create_test_panel_data()

        validator = PanelStructureValidator()
        config = DIDConfig(
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            panel=False,
        )

        result = validator.validate(df, config)
        assert result.is_valid
        assert any("panel=False was specified" in w for w in result.warnings)

    def test_treatment_reversibility(self):
        df = create_test_panel_data()
        treated_unit = df.filter(pl.col("g") > 0)["id"][0]
        df = df.with_columns(
            pl.when((pl.col("id") == treated_unit) & (pl.col("time") == 4)).then(0).otherwise(pl.col("g")).alias("g")
        )

        with pytest.raises(ValueError, match="must be irreversible"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
            )

    @pytest.mark.filterwarnings("ignore:.*units were already treated:UserWarning")
    @pytest.mark.filterwarnings("ignore:Dropped.*units:UserWarning")
    def test_early_treatment_handling(self):
        df = create_test_panel_data(n_periods=5)
        early_treated_units = df["id"].unique().to_list()[:10]
        df = df.with_columns(
            pl.when(pl.col("id").is_in(early_treated_units)).then(0.5).otherwise(pl.col("g")).alias("g")
        )

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
        )

        assert not result.data["id"].is_in(early_treated_units).any()


class TestCovariateHandling:
    def test_formula_parsing(self):
        df = create_test_panel_data().with_columns(pl.col("x1").alias("x.1"), pl.col("x2").alias("x 2"))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ x1 + x2",
        )

        assert result.covariates_tensor is not None
        assert all(cov.shape[1] == 3 for cov in result.covariates_tensor)

        renamed = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ x.1 + `x 2`",
        )

        for cov, cov_renamed in zip(result.covariates_tensor, renamed.covariates_tensor, strict=True):
            np.testing.assert_array_equal(cov, cov_renamed)


@pytest.mark.parametrize(
    "xformla, message",
    [
        ("~ x1 * x2", "xformla term 'x1 * x2' is not a column name"),
        ("~ x1 + x2 + x1:x2", "xformla term 'x1:x2' is not a column name"),
        ("~ C(x1)", "xformla term 'C(x1)' is not a column name"),
        ("~ x1 + I(x2**2)", "xformla term 'I(x2**2)' is not a column name"),
        ("~ 0 + x1 + x2", "xformla term '0' drops the intercept"),
    ],
)
def test_formula_rejects_terms_that_are_not_columns(xformla, message):
    df = create_test_panel_data()

    with pytest.raises(ValueError, match=re.escape(message)):
        preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla=xformla,
        )


class TestWeightHandling:
    def test_zero_weights(self):
        df = create_test_panel_data()
        w_vals = np.random.uniform(0.5, 2, len(df))
        w_vals[:20] = 0
        df = df.with_columns(pl.Series("w", w_vals))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            weightsname="w",
        )

        assert len(result.data) == len(df)
        assert (result.weights >= 0).all()
        assert (result.weights == 0).sum() == 5

    def test_weight_normalization(self):
        df = create_test_panel_data()
        df = df.with_columns(pl.Series("w", np.random.uniform(1, 10, len(df))))

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            weightsname="w",
        )

        assert np.isclose(result.weights.mean(), 1.0, rtol=0.02)


class TestUnbalancedPanelHandling:
    @pytest.mark.filterwarnings("ignore:.*units have unbalanced observations:UserWarning")
    @pytest.mark.filterwarnings("ignore:Dropped.*units while converting:UserWarning")
    def test_unbalanced_to_balanced_conversion(self):
        df = create_unbalanced_panel_data(missing_fraction=0.3)

        result_unbalanced = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            allow_unbalanced_panel=True,
        )

        assert not result_unbalanced.is_balanced_panel

        result_balanced = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            allow_unbalanced_panel=False,
        )

        assert result_balanced.is_balanced_panel
        assert result_balanced.config.id_count < df["id"].n_unique()

    def test_time_invariant_covariate_detection(self):
        df = create_test_panel_data()
        df = df.with_columns(
            [
                (pl.col("time") * pl.Series(np.random.normal(0, 1, len(df)))).alias("time_varying"),
            ]
        )
        first_x1 = df.group_by("id").agg(pl.col("x1").first().alias("time_invariant"))
        df = df.join(first_x1, on="id", how="left")

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            xformla="~ time_varying + time_invariant",
            allow_unbalanced_panel=True,
        )

        if not result.is_balanced_panel:
            assert "time_invariant" in result.time_invariant_data.columns
            assert "time_varying" not in result.time_invariant_data.columns


class TestClusteringOptions:
    def test_multiple_clustering_vars(self):
        df = create_test_panel_data()
        df = df.with_columns(
            [
                (pl.col("id") // 10).alias("cluster1"),
                (pl.col("time") % 2).alias("cluster2"),
            ]
        )

        with pytest.raises(ValueError, match="You can only provide 1 cluster variable"):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                clustervars=["cluster1", "cluster2"],
            )

        result = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            clustervars=["cluster1"],
        )

        assert result.cluster is not None
        assert result.config.clustervars == ["cluster1"]

    def test_invalid_cluster_var(self):
        df = create_test_panel_data()

        with pytest.raises(
            (ValueError, KeyError, pl.exceptions.ColumnNotFoundError),
            match="not found|Column not found|unable to find column",
        ):
            preprocess_did(
                data=df,
                yname="y",
                tname="time",
                idname="id",
                gname="g",
                clustervars=["nonexistent_var"],
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
        )

        assert result_universal.config.base_period == BasePeriod.UNIVERSAL

        result_varying = preprocess_did(
            data=df,
            yname="y",
            tname="time",
            idname="id",
            gname="g",
            base_period="varying",
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

        builder = PreprocessDataBuilder()
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

        builder = PreprocessDataBuilder()
        result = builder.with_data(df).with_config(config).validate().transform().build()

        assert isinstance(result, DIDData)
        assert result.config == config


@pytest.mark.parametrize(
    "param,value,match",
    [
        ("est_method", "ols", "est_method='ols' is not valid"),
        ("control_group", "all", "control_group='all' is not valid"),
        ("base_period", "fixed", "base_period='fixed' is not valid"),
    ],
)
def test_att_gt_invalid_params(param, value, match):
    df = create_test_panel_data()
    with pytest.raises(ValueError, match=match):
        att_gt(data=df, yname="y", tname="time", idname="id", gname="g", **{param: value})


def test_aggte_invalid_type():
    df = create_test_panel_data()
    result = att_gt(data=df, yname="y", tname="time", idname="id", gname="g")
    with pytest.raises(ValueError, match="type='invalid' is not valid"):
        aggte(result, type="invalid")


def test_column_validator_non_numeric_yname():
    df = create_test_panel_data()
    df = df.with_columns(pl.col("y").cast(pl.Utf8).alias("y_str"))
    validator = ColumnValidator()
    config = DIDConfig(yname="y_str", tname="time", idname="id", gname="g")
    result = validator.validate(df, config)
    assert not result.is_valid
    assert any("yname" in err and "not numeric" in err for err in result.errors)


def test_column_validator_missing_covariate():
    df = create_test_panel_data()
    validator = ColumnValidator()
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", xformla="~ x1 + missing_var")
    result = validator.validate(df, config)
    assert not result.is_valid
    assert any("missing_var" in err for err in result.errors)


@pytest.mark.parametrize("weights", ["negative", "zero"])
def test_weight_normalizer_rejects_negative_weights_and_a_zero_mean(weights):
    df = create_test_panel_data()
    values = np.random.uniform(-1, 1, len(df)) if weights == "negative" else np.zeros(len(df))
    df = df.with_columns(pl.Series("w", values))
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", weightsname="w")
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=re.escape(message)):
        WeightNormalizer().transform(df, config)


def test_argument_validator_invalid_biters():
    df = create_test_panel_data()
    validator = ArgumentValidator()
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", biters=0)
    result = validator.validate(df, config)
    assert result.is_valid


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
@pytest.mark.parametrize("allow_unbalanced_panel", [False, True])
def test_preprocess_did_panel_false_keys_every_row(mpdta_data, allow_unbalanced_panel):
    result = preprocess_did(
        mpdta_data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        panel=False,
        allow_unbalanced_panel=allow_unbalanced_panel,
    )

    assert result.config.true_repeated_cross_sections is True
    assert result.config.idname == ".rowid"
    assert result.config.id_count == mpdta_data.height
    assert result.time_invariant_data.height == mpdta_data.height


def test_preprocess_did_unbalanced_panel_keeps_unit_keys(mpdta_unbalanced):
    result = preprocess_did(
        mpdta_unbalanced,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
    )

    assert result.config.panel is False
    assert result.config.true_repeated_cross_sections is False
    assert result.config.idname == "countyreal"
    assert result.config.id_count == mpdta_unbalanced["countyreal"].n_unique()
    assert result.time_invariant_data.height == result.config.id_count


def test_preprocess_did_keeps_nan_cohort_out_of_never_treated(mpdta_nan_cohort):
    with pytest.warns(UserWarning, match="^Dropped 50 rows from original data due to missing values$"):
        result = preprocess_did(mpdta_nan_cohort, yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    cohorts = result.time_invariant_data["first.treat"]

    assert result.config.id_count == 490
    assert cohorts.is_infinite().sum() == 309
    assert cohorts.is_nan().sum() == 0


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (DIDConfig(yname="y", tname="time", gname="g"), [np.inf, np.inf, np.inf]),
        (DIDConfig(yname="y", tname="time", gname="g", anticipation=1), [np.inf, 4.0, np.inf]),
        (ContDIDConfig(yname="y", tname="time", gname="g", anticipation=1), [np.inf, np.inf, np.inf]),
    ],
)
def test_treatment_encoder_codes_never_treated_cohorts(cohort_codes_panel, config, expected):
    result = TreatmentEncoder().transform(cohort_codes_panel, config)

    assert result.filter(pl.col("time") == 1)["g"].to_list() == expected


@pytest.mark.filterwarnings("ignore:No never-treated group is available:UserWarning")
@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_preprocess_did_without_never_treated_treats_earlier_cohorts_only(mpdta_without_never_treated, control_group):
    result = preprocess_did(
        mpdta_without_never_treated,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group=control_group,
    )

    np.testing.assert_array_equal(result.config.treated_groups, [2004.0, 2006.0])
    np.testing.assert_array_equal(result.config.time_periods, [2003, 2004, 2005, 2006])
    assert result.config.id_count == 191


def test_preprocess_did_weights_tensor_holds_each_period(mpdta_varying_weights):
    result = preprocess_did(
        mpdta_varying_weights, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="w"
    )
    normalized = mpdta_varying_weights.with_columns(pl.col("w") / pl.col("w").mean())
    units = result.time_invariant_data["countyreal"]

    assert len(result.weights_tensor) == 5
    for period, weights in zip(result.config.time_periods, result.weights_tensor):
        rows = normalized.filter(pl.col("year") == period)
        np.testing.assert_allclose(weights, units.replace_strict(rows["countyreal"], rows["w"]).to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("panel", [True, False])
def test_preprocess_did_weights_tensor_needs_balanced_panel(mpdta_unbalanced_varying_weights, panel):
    result = preprocess_did(
        mpdta_unbalanced_varying_weights,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="w",
        panel=panel,
    )

    assert result.weights_tensor is None


@pytest.mark.parametrize("reserved", [WEIGHTS_COLUMN, ROW_ID_COLUMN])
def test_column_validator_rejects_reserved_names(reserved):
    df = create_test_panel_data().rename({"y": reserved, "x1": f"{reserved}x"})
    config = DIDConfig(yname=reserved, tname="time", idname="id", gname="g", xformla=f"~ {reserved}x")

    result = ColumnValidator().validate(df, config)

    assert not result.is_valid
    assert result.errors == [
        f"yname names the column '{reserved}'. "
        "Since moderndid uses that name for an internal column, rename the column."
    ]


@pytest.mark.parametrize(
    "pairs, where",
    [
        ([(1, 1)], "the (id, time) pair (1, 1)"),
        ([(2, 2), (1, 1)], "the (id, time) pairs (1, 1) and (2, 2)"),
        ([(3, 1), (1, 1), (2, 2)], "the (id, time) pairs (1, 1), (2, 2), and (3, 1)"),
        ([(5, 1), (4, 2), (3, 1), (2, 2), (1, 1)], "5 (id, time) pairs, such as (1, 1), (2, 2), and (3, 1)"),
    ],
)
def test_panel_structure_validator_names_repeated_unit_periods(unit_period_panel, pairs, where):
    repeated = pl.concat([unit_period_panel.filter((pl.col("id") == i) & (pl.col("time") == t)) for i, t in pairs])
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=True)

    result = PanelStructureValidator().validate(pl.concat([unit_period_panel, repeated]), config)

    assert result.errors == [
        "The value of idname must be unique (by tname). Some units are observed more than once in a period. "
        f"Rows repeat for {where}."
    ]
    assert result.warnings == []


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_panel_structure_validator_skips_rows_missing_unit_or_period(unit_period_panel, missing):
    panel = unit_period_panel.with_columns(pl.col("id", "time").cast(pl.Float64))
    incomplete = pl.DataFrame(
        {"id": [1.0, 1.0, missing, missing], "time": [missing, missing, 2.0, 2.0], "y": [0.0] * 4, "g": [0] * 4},
        schema=panel.schema,
    )
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=True, allow_unbalanced_panel=True)

    result = PanelStructureValidator().validate(pl.concat([panel, incomplete]), config)

    assert result.errors == []


def test_create_tensors_rejects_unit_major_rows(mpdta_data):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    dp = PreprocessDataBuilder().with_data(mpdta_data).with_config(config).validate().transform().build()
    message = (
        "The panel tensors take each period's rows by position. The data must hold one block of rows per period "
        "and list the units in the same order in every block. Sort it by period, cohort, and unit."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        TensorFactorySelector.create_tensors(dp.data.sort("countyreal", "year"), dp.config)


def test_create_tensors_rejects_units_reordered_within_one_period(mpdta_data):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    dp = PreprocessDataBuilder().with_data(mpdta_data).with_config(config).validate().transform().build()
    in_2005 = pl.col("year") == 2005
    blocks = [dp.data.filter(~in_2005), dp.data.filter(in_2005).reverse()]
    reordered = pl.concat(blocks).sort("year", maintain_order=True)

    with pytest.raises(ValueError, match="list the units in the same order in every block"):
        TensorFactorySelector.create_tensors(reordered, dp.config)


def test_create_tensors_pairs_each_unit_across_periods_on_shuffled_rows(mpdta_shuffled):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    dp = PreprocessDataBuilder().with_data(mpdta_shuffled).with_config(config).validate().transform().build()
    wide = dp.time_invariant_data.select("countyreal").join(
        mpdta_shuffled.pivot(on="year", index="countyreal", values="lemp"), on="countyreal", how="left"
    )

    for i, year in enumerate(range(2003, 2008)):
        np.testing.assert_array_equal(dp.outcomes_tensor[i], wide[str(year)].to_numpy())
