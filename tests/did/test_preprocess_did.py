"""Tests for DiD preprocessing functions."""

import re
import warnings

import numpy as np
import pytest

from tests.helpers import importorskip

pl = importorskip("polars")

from moderndid import aggte, att_gt, drdid_panel, preprocess_did
from moderndid.core.preprocess import (
    ROW_ID_COLUMN,
    WEIGHTS_COLUMN,
    BasePeriod,
    CompositeValidator,
    ContDIDConfig,
    ControlGroup,
    DataFormat,
    DataTransformerPipeline,
    DIDConfig,
    DIDData,
    EstimationMethod,
    PreprocessDataBuilder,
    TensorFactorySelector,
)
from moderndid.core.preprocess.transformers import TreatmentEncoder, WeightNormalizer
from moderndid.core.preprocess.validators import (
    ArgumentValidator,
    ClusterValidator,
    ColumnValidator,
    PanelStructureValidator,
    TreatmentValidator,
    check_columns,
)


def test_column_validator(mpdta_data):
    present = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    missing = DIDConfig(yname="employment", tname="year", idname="countyreal", gname="first.treat")

    valid = ColumnValidator().validate(mpdta_data, present)
    invalid = ColumnValidator().validate(mpdta_data, missing)

    assert valid.is_valid
    assert valid.errors == []
    assert not invalid.is_valid
    assert invalid.errors == ["yname='employment' is not a column in the data."]


def test_treatment_validator(mpdta_data, mpdta_cohort_switched_off):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    message = (
        "The value of gname (treatment variable) must be the same across all periods for each particular unit. "
        "The treatment must be irreversible."
    )

    valid = TreatmentValidator().validate(mpdta_data, config)
    invalid = TreatmentValidator().validate(mpdta_cohort_switched_off, config)

    assert valid.is_valid
    assert valid.errors == []
    assert not invalid.is_valid
    assert invalid.errors == [message]


def test_treatment_validator_lets_the_cohort_change_across_rows_without_a_panel(mpdta_cohort_switched_off):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat", panel=False)

    result = TreatmentValidator().validate(mpdta_cohort_switched_off, config)

    assert result.errors == []


def test_argument_validator(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}
    accepted = DIDConfig(
        control_group=ControlGroup.NOT_YET_TREATED, base_period="universal", est_method=EstimationMethod.IPW, **spec
    )
    rejected = DIDConfig(control_group="all", base_period="fixed", est_method="ols", **spec)

    valid = ArgumentValidator().validate(mpdta_data, accepted)
    invalid = ArgumentValidator().validate(mpdta_data, rejected)

    assert valid.errors == []
    assert invalid.errors == [
        "control_group='all' is not valid. Must be 'nevertreated' or 'notyettreated'.",
        "base_period='fixed' is not valid. Must be 'varying' or 'universal'.",
        "est_method='ols' is not valid. Must be 'dr', 'ipw', or 'reg'.",
    ]


def test_composite_validator(mpdta_data):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}
    string_periods = mpdta_data.with_columns(pl.col("year").cast(pl.Utf8))

    columns = CompositeValidator().validate(string_periods, DIDConfig(control_group="all", **spec))
    structure = CompositeValidator(phase="structure").validate(mpdta_data, DIDConfig(panel=False, **spec))

    assert not columns.is_valid
    assert columns.errors == [
        "control_group='all' is not valid. Must be 'nevertreated' or 'notyettreated'.",
        "tname = 'year' is not numeric. Please convert it",
    ]
    assert structure.is_valid
    assert structure.errors == []
    assert structure.warnings == [
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator."
    ]


def test_weight_normalizer(weighted_unit_period_panel):
    spec = {"yname": "y", "tname": "time", "idname": "id", "gname": "g"}

    weighted = WeightNormalizer().transform(weighted_unit_period_panel, DIDConfig(weightsname="w", **spec))
    unweighted = WeightNormalizer().transform(weighted_unit_period_panel, DIDConfig(**spec))

    np.testing.assert_array_equal(
        weighted[WEIGHTS_COLUMN].to_numpy(), [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.0, 1.0]
    )
    np.testing.assert_array_equal(unweighted[WEIGHTS_COLUMN].to_numpy(), np.ones(10))


def test_treatment_encoder_keeps_a_cohort_that_starts_in_the_last_period(cohort_codes_panel):
    data = cohort_codes_panel.with_columns(pl.Series("g", [0, 0, 3, 3, 2, 2]))
    config = DIDConfig(yname="y", tname="time", gname="g")

    result = TreatmentEncoder().transform(data, config)

    assert result["g"].to_list() == [np.inf, np.inf, 3.0, 3.0, 2.0, 2.0]


def test_transformer_pipeline(weighted_unit_period_panel):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", weightsname="w")

    result = DataTransformerPipeline.get_did_pipeline().transform(weighted_unit_period_panel, config)

    assert result.select("time", "g", "id").rows() == [
        (1, 2.0, 2),
        (1, 2.0, 4),
        (1, np.inf, 1),
        (1, np.inf, 3),
        (1, np.inf, 5),
        (2, 2.0, 2),
        (2, 2.0, 4),
        (2, np.inf, 1),
        (2, np.inf, 3),
        (2, np.inf, 5),
    ]
    np.testing.assert_array_equal(result[WEIGHTS_COLUMN].to_numpy(), [1.5, 1.5, 0.5, 0.5, 1.0] * 2)
    np.testing.assert_array_equal(config.time_periods, [1, 2])
    np.testing.assert_array_equal(config.treated_groups, [2.0])
    assert config.id_count == 5
    assert config.data_format == DataFormat.PANEL


def test_panel_tensor_factory(weighted_unit_period_panel):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", xformla="~ x", weightsname="w")
    data = DataTransformerPipeline.get_did_pipeline().transform(weighted_unit_period_panel, config)

    tensors = TensorFactorySelector.create_tensors(data, config)

    np.testing.assert_array_equal(tensors["outcomes_tensor"], [[0.2, 0.6, 0.1, 0.3, 0.2], [0.9, 1.1, 0.4, 0.5, 0.8]])
    np.testing.assert_array_equal(
        tensors["covariates_tensor"],
        [
            [[1.0, -1.0], [1.0, 4.0], [1.0, 0.5], [1.0, 0.0], [1.0, 1.0]],
            [[1.0, 2.0], [1.0, -2.0], [1.0, 1.5], [1.0, 3.0], [1.0, 1.0]],
        ],
    )
    np.testing.assert_array_equal(tensors["weights_tensor"], [[1.5, 1.5, 0.5, 0.5, 1.0]] * 2)
    np.testing.assert_array_equal(tensors["weights"], [1.5, 1.5, 0.5, 0.5, 1.0])
    assert tensors["time_invariant_data"].rows() == [
        (2, 2.0, 1.5),
        (4, 2.0, 1.5),
        (1, np.inf, 0.5),
        (3, np.inf, 0.5),
        (5, np.inf, 1.0),
    ]
    assert tensors["covariates_matrix"] is None


def test_rcs_tensor_factory(weighted_unit_period_panel):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", xformla="~ x", panel=False)
    data = DataTransformerPipeline.get_did_pipeline().transform(weighted_unit_period_panel, config)

    tensors = TensorFactorySelector.create_tensors(data, config)

    assert tensors["data"][ROW_ID_COLUMN].to_list() == [2, 6, 0, 4, 8, 3, 7, 1, 5, 9]
    np.testing.assert_array_equal(
        tensors["covariates_matrix"],
        [
            [1.0, -1.0],
            [1.0, 4.0],
            [1.0, 0.5],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, -2.0],
            [1.0, 1.5],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
    )
    assert tensors["outcomes_tensor"] is None
    assert tensors["covariates_tensor"] is None


def test_basic_preprocessing(mpdta_data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = preprocess_did(mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")

    assert [str(warning.message) for warning in caught] == []
    assert result.config.data_format == DataFormat.PANEL
    np.testing.assert_array_equal(result.config.time_periods, [2003, 2004, 2005, 2006, 2007])
    np.testing.assert_array_equal(result.config.treated_groups, [2004, 2006, 2007])
    assert result.config.id_count == 500
    assert result.cohort_counts.sort("cohort").rows() == [(2004.0, 20), (2006.0, 40), (2007.0, 131), (np.inf, 309)]
    assert result.period_counts.sort("period").rows() == [(year, 500) for year in range(2003, 2008)]
    assert result.crosstable_counts.columns == ["period", "2004.0", "2006.0", "2007.0", "inf"]
    assert result.crosstable_counts.sort("period").rows() == [(year, 20, 40, 131, 309) for year in range(2003, 2008)]
    np.testing.assert_array_equal(result.weights, np.ones(500))


def test_with_all_options(mpdta_varying_weights_and_states):
    with pytest.warns(UserWarning, match="^Dropped 20 units that were already treated in the first period$"):
        result = preprocess_did(
            data=mpdta_varying_weights_and_states,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            xformla="~ lpop",
            panel=True,
            allow_unbalanced_panel=False,
            control_group="notyettreated",
            anticipation=1,
            weightsname="w",
            clustervars=["countyreal", "state"],
            est_method="ipw",
            base_period="universal",
        )

    assert result.config.control_group == ControlGroup.NOT_YET_TREATED
    assert result.config.anticipation == 1
    assert result.config.weightsname == "w"
    assert result.config.est_method == EstimationMethod.IPW
    assert result.config.base_period == BasePeriod.UNIVERSAL
    assert result.config.clustervars == ["countyreal", "state"]
    assert result.config.id_count == 480
    np.testing.assert_array_equal(result.cluster, result.time_invariant_data["countyreal"].to_numpy() // 1000)


def test_repeated_cross_section_preprocessing(mpdta_data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = preprocess_did(
            mpdta_data.drop("countyreal"), yname="lemp", tname="year", gname="first.treat", panel=False
        )

    assert [str(warning.message) for warning in caught] == []
    assert result.config.true_repeated_cross_sections is True
    assert result.config.idname == ROW_ID_COLUMN
    assert result.config.data_format == DataFormat.REPEATED_CROSS_SECTION
    assert result.config.id_count == 2500
    np.testing.assert_array_equal(result.data[ROW_ID_COLUMN].sort().to_numpy(), np.arange(2500))
    np.testing.assert_array_equal(result.covariates_matrix, np.ones((2500, 1)))
    assert result.outcomes_tensor is None


@pytest.mark.parametrize(
    ("control_group", "n_never_treated", "messages"),
    [
        (
            "nevertreated",
            131,
            ["No never-treated group is available. The last treated cohort is being coerced as 'never-treated' units."],
        ),
        ("notyettreated", 0, []),
    ],
)
def test_no_never_treated(mpdta_without_never_treated, control_group, n_never_treated, messages):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = preprocess_did(
            mpdta_without_never_treated,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            control_group=control_group,
        )
    cohorts = result.time_invariant_data["first.treat"]

    assert [str(warning.message) for warning in caught] == messages
    assert cohorts.is_infinite().sum() == n_never_treated
    assert (cohorts == 2007).sum() == 131 - n_never_treated


def test_empty_groups_error(mpdta_data):
    data = mpdta_data.with_columns(pl.lit(2003).alias("first.treat"))
    message = (
        "No valid groups. The variable in 'gname' should be expressed as the time a unit is first treated "
        "(0 if never-treated)"
    )

    with (
        pytest.warns(UserWarning, match="^Dropped 500 units that were already treated in the first period$"),
        pytest.raises(ValueError, match=f"^{re.escape(message)}$"),
    ):
        preprocess_did(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")


def test_invalid_control_group(mpdta_data):
    with pytest.raises(
        ValueError,
        match=(
            "^Validation failed:\ncontrol_group='invalid' is not valid\\. Must be 'nevertreated' or 'notyettreated'\\.$"
        ),
    ):
        preprocess_did(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            control_group="invalid",
        )


def test_missing_data_handling(mpdta_data):
    missing = (pl.col("countyreal") == 8001) & (pl.col("year") >= 2006)
    data = mpdta_data.with_columns(pl.when(missing).then(float("nan")).otherwise(pl.col("lemp")).alias("lemp"))

    with pytest.warns(UserWarning, match="^Dropped 2 rows from original data due to missing values$"):
        result = preprocess_did(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")

    assert result.data.height == 2498
    assert result.data.filter(pl.col("countyreal") == 8001)["year"].to_list() == [2003, 2004, 2005]


def test_string_time_periods(mpdta_data):
    data = mpdta_data.with_columns(pl.col("year").cast(pl.Utf8))
    message = "Validation failed:\ntname = 'year' is not numeric. Please convert it"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        preprocess_did(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")


def test_single_treated_unit(mpdta_data):
    data = mpdta_data.filter((pl.col("first.treat") != 2004) | (pl.col("countyreal") == 17005))
    message = "Be aware that there are some small groups in your dataset.\nCheck groups: 2004"

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$"):
        result = preprocess_did(data, yname="lemp", tname="year", idname="countyreal", gname="first.treat")

    np.testing.assert_array_equal(result.config.treated_groups, [2004, 2006, 2007])
    assert result.config.id_count == 481


def test_panel_true_with_rcs_data(rcs_df):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=True)

    result = PanelStructureValidator().validate(rcs_df, config)

    assert not result.is_valid
    assert result.errors == [
        "panel=True was specified, but no units appear in multiple time periods. "
        "Your data appears to be repeated cross-sections. "
        "Set panel=False to use the repeated cross-section estimator."
    ]
    assert result.warnings == []


def test_panel_false_with_panel_data(panel_df):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=False)

    result = PanelStructureValidator().validate(panel_df, config)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == [
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator."
    ]


def test_panel_structure_validator_allows_repeated_unit_periods_without_a_panel(unit_period_panel):
    repeated = pl.concat([unit_period_panel, unit_period_panel.head(1)])
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=False)

    result = PanelStructureValidator().validate(repeated, config)

    assert result.errors == []


def test_treatment_reversibility(mpdta_cohort_switched_off):
    with pytest.raises(ValueError, match="must be irreversible"):
        preprocess_did(
            data=mpdta_cohort_switched_off,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
        )


def test_early_treatment_handling(mpdta_early_cohorts):
    with pytest.warns(UserWarning, match="^Dropped 10 units that were already treated in the first period$"):
        result = preprocess_did(
            mpdta_early_cohorts, yname="lemp", tname="year", idname="countyreal", gname="first.treat"
        )
    cohorts = result.time_invariant_data["first.treat"]

    assert result.config.id_count == 490
    assert result.time_invariant_data.filter(pl.col("countyreal") < 9000).height == 0
    assert (cohorts == 2004).sum() == 33


def test_preprocess_did_drops_rows_treated_in_the_first_period_without_idname(mpdta_early_cohorts):
    data = mpdta_early_cohorts.drop("countyreal")

    with pytest.warns(UserWarning, match="^Dropped 50 observations that were already treated in the first period$"):
        result = preprocess_did(data, yname="lemp", tname="year", gname="first.treat", panel=False)

    assert result.config.id_count == 2450
    assert result.data.filter(pl.col("first.treat") == 2003).height == 0


def test_formula_parsing(mpdta_data):
    data = mpdta_data.with_columns((pl.col("lpop") + 0.1 * (pl.col("year") - 2003)).alias("growing")).with_columns(
        pl.col("lpop").alias("log.pop"), pl.col("growing").alias("growing pop")
    )
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}

    plain = preprocess_did(data, xformla="~ lpop + growing", **spec)
    renamed = preprocess_did(data, xformla="~ log.pop + `growing pop`", **spec)
    units = plain.time_invariant_data.select("countyreal")
    by_year = [
        units.join(data.filter(pl.col("year") == year), on="countyreal", how="left") for year in range(2003, 2008)
    ]
    expected = np.stack([np.column_stack([np.ones(500), rows["lpop"], rows["growing"]]) for rows in by_year])

    np.testing.assert_array_equal(np.stack(plain.covariates_tensor), expected)
    np.testing.assert_array_equal(np.stack(renamed.covariates_tensor), expected)


@pytest.mark.parametrize(
    "xformla, message",
    [
        ("~ lpop * treat", "xformla term 'lpop * treat' is not a column name"),
        ("~ lpop + treat + lpop:treat", "xformla term 'lpop:treat' is not a column name"),
        ("~ C(treat)", "xformla term 'C(treat)' is not a column name"),
        ("~ lpop + I(lpop**2)", "xformla term 'I(lpop**2)' is not a column name"),
        ("~ 0 + lpop + treat", "xformla term '0' drops the intercept"),
    ],
)
def test_formula_rejects_terms_that_are_not_columns(mpdta_data, xformla, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        preprocess_did(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            xformla=xformla,
        )


def test_zero_weights(mpdta_pop_weighted):
    zeroed = mpdta_pop_weighted["countyreal"].unique().sort().head(5)
    data = mpdta_pop_weighted.with_columns(
        pl.when(pl.col("countyreal").is_in(zeroed.implode())).then(0.0).otherwise(pl.col("pop")).alias("pop")
    )

    result = preprocess_did(
        data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="pop"
    )

    assert result.config.id_count == 500
    np.testing.assert_array_equal(result.weights == 0, result.time_invariant_data["countyreal"].is_in(zeroed.implode()))


def test_weight_normalization(mpdta_pop_weighted):
    result = preprocess_did(
        mpdta_pop_weighted, yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="pop"
    )
    units = result.time_invariant_data.select("countyreal")
    pop = units.join(mpdta_pop_weighted.filter(pl.col("year") == 2003), on="countyreal", how="left")["pop"].to_numpy()

    np.testing.assert_allclose(result.weights, pop / pop.mean(), rtol=1e-12, atol=0)


def test_unbalanced_to_balanced_conversion(mpdta_unbalanced):
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}

    unbalanced = preprocess_did(mpdta_unbalanced, allow_unbalanced_panel=True, **spec)
    with pytest.warns(UserWarning, match="^Dropped 74 units while converting to balanced panel$"):
        balanced = preprocess_did(mpdta_unbalanced, allow_unbalanced_panel=False, **spec)

    assert not unbalanced.is_balanced_panel
    assert unbalanced.config.id_count == 500
    assert balanced.is_balanced_panel
    assert balanced.config.id_count == 426
    assert balanced.time_invariant_data.filter(pl.col("countyreal") % 7 == 0).height == 0


def test_preprocess_did_names_the_option_when_balancing_drops_every_unit(mpdta_rotating):
    message = (
        "All observations dropped while converting to balanced panel. "
        "Consider setting panel=False and/or revisiting 'idname'"
    )

    with (
        pytest.warns(UserWarning, match="^Dropped 500 units while converting to balanced panel$"),
        pytest.raises(ValueError, match=f"^{re.escape(message)}$"),
    ):
        preprocess_did(
            mpdta_rotating,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            allow_unbalanced_panel=False,
        )


@pytest.mark.parametrize("data_fixture", ["mpdta_data", "mpdta_unbalanced"])
def test_time_invariant_covariate_detection(request, data_fixture):
    data = request.getfixturevalue(data_fixture).with_columns((pl.col("lpop") + pl.col("year") - 2003).alias("growing"))

    result = preprocess_did(
        data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", xformla="~ growing + lpop"
    )

    assert result.time_invariant_data.columns == ["countyreal", "first.treat", WEIGHTS_COLUMN, "lpop"]


def test_multiple_clustering_vars(mpdta_with_states):
    data = mpdta_with_states.with_columns((pl.col("countyreal") % 10).alias("digit"))
    spec = {"yname": "lemp", "tname": "year", "idname": "countyreal", "gname": "first.treat"}
    message = "Validation failed:\nYou can only provide 1 cluster variable additionally to the one provided in idname"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        preprocess_did(data, clustervars=["state", "digit"], **spec)
    result = preprocess_did(data, clustervars=["state"], **spec)

    assert result.config.clustervars == ["state"]
    np.testing.assert_array_equal(result.cluster, result.time_invariant_data["countyreal"].to_numpy() // 1000)


def test_preprocess_did_rejects_a_cluster_that_changes_within_a_unit(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") + pl.col("year")).alias("cluster"))
    message = (
        "Validation failed:\nDiD cannot handle time-varying cluster variables at the moment. "
        "Please check your cluster variable."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        preprocess_did(
            data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", clustervars=["cluster"]
        )


def test_cluster_validator_lets_the_cluster_change_across_rows_without_a_panel(mpdta_data):
    data = mpdta_data.with_columns((pl.col("countyreal") + pl.col("year")).alias("cluster"))
    config = DIDConfig(
        yname="lemp", tname="year", idname="countyreal", gname="first.treat", clustervars=["cluster"], panel=False
    )

    result = ClusterValidator().validate(data, config)

    assert result.errors == []


def test_invalid_cluster_var(mpdta_data):
    message = "Validation failed:\n'nonexistent_var' in clustervars is not a column in the data."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        preprocess_did(
            data=mpdta_data,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            clustervars=["nonexistent_var"],
        )


def test_anticipation_effects(mpdta_without_never_treated):
    with pytest.warns(UserWarning, match="^Dropped 20 units that were already treated in the first period$"):
        result = preprocess_did(
            mpdta_without_never_treated,
            yname="lemp",
            tname="year",
            idname="countyreal",
            gname="first.treat",
            anticipation=1,
            control_group="notyettreated",
        )

    assert result.config.anticipation == 1
    np.testing.assert_array_equal(result.config.time_periods, [2003, 2004, 2005])
    np.testing.assert_array_equal(result.config.treated_groups, [2006])
    assert result.config.id_count == 171


@pytest.mark.parametrize(
    ("option", "value", "member"),
    [
        ("control_group", "notyettreated", ControlGroup.NOT_YET_TREATED),
        ("base_period", "universal", BasePeriod.UNIVERSAL),
        ("base_period", "varying", BasePeriod.VARYING),
        ("est_method", "dr", EstimationMethod.DOUBLY_ROBUST),
        ("est_method", "ipw", EstimationMethod.IPW),
        ("est_method", "reg", EstimationMethod.REGRESSION),
    ],
)
def test_preprocess_did_stores_each_option_as_its_enum_member(mpdta_data, option, value, member):
    result = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **{option: value}
    )

    assert getattr(result.config, option) is member


def test_builder_validation_errors(mpdta_data):
    config = DIDConfig(yname="missing_column", tname="year", idname="countyreal", gname="first.treat")
    message = "Validation failed:\nyname='missing_column' is not a column in the data."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        PreprocessDataBuilder().with_data(mpdta_data).with_config(config).validate()


def test_builder_chaining(mpdta_data):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    builder = PreprocessDataBuilder()

    steps = [builder.with_data(mpdta_data), builder.with_config(config), builder.validate(), builder.transform()]
    result = builder.build()

    assert all(step is builder for step in steps)
    assert isinstance(result, DIDData)
    assert result.config.id_count == 500


@pytest.mark.parametrize(
    "param,value,match",
    [
        ("est_method", "ols", "est_method='ols' is not valid"),
        ("control_group", "all", "control_group='all' is not valid"),
        ("base_period", "fixed", "base_period='fixed' is not valid"),
    ],
)
def test_att_gt_invalid_params(mpdta_data, param, value, match):
    with pytest.raises(ValueError, match=match):
        att_gt(data=mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **{param: value})


def test_aggte_invalid_type(att_gt_baseline_result):
    with pytest.raises(ValueError, match="type='invalid' is not valid"):
        aggte(att_gt_baseline_result, type="invalid")


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("year", "tname = 'year' is not numeric. Please convert it"),
        ("first.treat", "gname = 'first.treat' is not numeric. Please convert it"),
        ("countyreal", "idname = 'countyreal' is not numeric. Please convert it"),
        ("lemp", "yname = 'lemp' is not numeric. Please convert it"),
    ],
)
def test_column_validator_rejects_a_column_that_is_not_numeric(mpdta_data, column, message):
    data = mpdta_data.with_columns(pl.col(column).cast(pl.Utf8))
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")

    result = ColumnValidator().validate(data, config)

    assert not result.is_valid
    assert result.errors == [message]


def test_column_validator_missing_covariate(mpdta_data):
    config = DIDConfig(
        yname="lemp", tname="year", idname="countyreal", gname="first.treat", xformla="~ lpop + missing_var"
    )

    result = ColumnValidator().validate(mpdta_data, config)

    assert not result.is_valid
    assert result.errors == ["'missing_var' in xformla is not a column in the data."]


@pytest.mark.parametrize("weights", ["negative", "zero"])
def test_weight_normalizer_rejects_negative_weights_and_a_zero_mean(mpdta_data, weights):
    first_row = np.arange(mpdta_data.height) == 0
    values = np.where(first_row, -1.0, 1.0) if weights == "negative" else np.zeros(mpdta_data.height)
    data = mpdta_data.with_columns(pl.Series("w", values))
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat", weightsname="w")
    message = "The weights variable 'w' must be non-negative with a positive mean."

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        WeightNormalizer().transform(data, config)


@pytest.mark.parametrize("argument", [{"alpha": 1.5}, {"anticipation": -1}, {"biters": 0}])
def test_argument_validator_leaves_alpha_anticipation_and_biters_to_the_estimator(mpdta_data, argument):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat", **argument)

    result = ArgumentValidator().validate(mpdta_data, config)

    assert result.is_valid
    assert result.errors == []


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


def test_preprocess_did_stores_alpha(mpdta_data):
    result = preprocess_did(mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", alpha=0.1)

    assert result.config.alpha == 0.1


@pytest.mark.parametrize(
    ("argument", "field", "value"),
    [
        ("boot", "boot", True),
        ("cband", "cband", True),
        ("biters", "biters", 99),
        ("faster_mode", "faster_mode", True),
        ("pl_parallel", "pl", True),
        ("cores", "cores", 2),
    ],
)
def test_preprocess_did_stores_the_bootstrap_and_parallel_settings(mpdta_data, argument, field, value):
    result = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", **{argument: value}
    )

    assert getattr(result.config, field) == value


def test_preprocess_did_rejects_the_removed_alp_argument(mpdta_data):
    with pytest.raises(TypeError, match="unexpected keyword argument 'alp'"):
        preprocess_did(mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", alp=0.1)


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


@pytest.mark.filterwarnings("ignore:panel=False was specified:UserWarning")
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
def test_column_validator_rejects_reserved_names(mpdta_data, reserved):
    data = mpdta_data.rename({"lemp": reserved, "lpop": f"{reserved}x"})
    config = DIDConfig(yname=reserved, tname="year", idname="countyreal", gname="first.treat", xformla=f"~ {reserved}x")

    result = ColumnValidator().validate(data, config)

    assert not result.is_valid
    assert result.errors == [
        f"yname names the column '{reserved}'. "
        "Since moderndid uses that name for an internal column, rename the column."
    ]


def test_column_validator_checks_idname_without_a_panel(mpdta_data):
    config = DIDConfig(yname="lemp", tname="year", idname="county", gname="first.treat", panel=False)

    result = ColumnValidator().validate(mpdta_data, config)

    assert result.errors == ["idname='county' is not a column in the data. Did you mean 'countyreal'?"]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"tname": "Year"}, "tname='Year' is not a column in the data. Did you mean 'year'?"),
        ({"weightsname": "population"}, "weightsname='population' is not a column in the data."),
        (
            {"clustervars": ["countyreal", "county"]},
            "'county' in clustervars is not a column in the data. Did you mean 'countyreal'?",
        ),
        (
            {"xformla": "~ lpop + I(lemp**2) + lpopp"},
            "'lpopp' in xformla is not a column in the data. Did you mean 'lpop'?",
        ),
        ({"vcov": {"CRV1": "countyreal+yeer"}}, "'yeer' in vcov is not a column in the data. Did you mean 'year'?"),
        (
            {"yname": "lemp", "tname": "yeer", "idname": None, "gname": "first_treat"},
            "tname='yeer' is not a column in the data. Did you mean 'year'?\n"
            "gname='first_treat' is not a column in the data. Did you mean 'first.treat' or 'treat'?",
        ),
    ],
)
def test_check_columns_names_each_missing_column(mpdta_data, arguments, message):
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        check_columns(mpdta_data, **arguments)


def test_check_columns_suggests_at_most_three_columns():
    data = pl.DataFrame({"cov1": [1.0], "cov2": [1.0], "cov3": [1.0], "cov4": [1.0]})
    message = "'cov5' in xformla is not a column in the data. Did you mean 'cov1', 'cov2', or 'cov3'?"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        check_columns(data, xformla="~ cov1 + cov5")


def test_check_columns_accepts_columns_that_exist(mpdta_data):
    result = check_columns(
        mpdta_data.to_pandas(),
        yname="lemp",
        tname="year",
        idname=None,
        clustervars=["countyreal"],
        xformla="~ lpop + I(lpop**2)",
        vcov={"CRV1": "countyreal+year"},
    )

    assert result is None


def test_preprocess_did_names_a_misspelled_column(mpdta_data):
    message = "Validation failed:\ntname='yeer' is not a column in the data. Did you mean 'year'?"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        preprocess_did(mpdta_data, yname="lemp", tname="yeer", idname="countyreal", gname="first.treat")


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

    assert not result.is_valid
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


def test_create_tensors_rejects_periods_out_of_order(mpdta_data):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    dp = PreprocessDataBuilder().with_data(mpdta_data).with_config(config).validate().transform().build()
    message = (
        "The panel tensors take each period's rows by position. The data must hold one block of rows per period "
        "and list the units in the same order in every block. Sort it by period, cohort, and unit."
    )

    with pytest.raises(ValueError, match=re.escape(message)):
        TensorFactorySelector.create_tensors(dp.data.sort("year", descending=True, maintain_order=True), dp.config)


def test_create_tensors_pairs_each_unit_across_periods_on_shuffled_rows(mpdta_shuffled):
    config = DIDConfig(yname="lemp", tname="year", idname="countyreal", gname="first.treat")
    dp = PreprocessDataBuilder().with_data(mpdta_shuffled).with_config(config).validate().transform().build()
    wide = dp.time_invariant_data.select("countyreal").join(
        mpdta_shuffled.pivot(on="year", index="countyreal", values="lemp"), on="countyreal", how="left"
    )
    expected = wide.select([str(year) for year in range(2003, 2008)]).to_numpy().T

    np.testing.assert_array_equal(np.stack(dp.outcomes_tensor), expected)


def test_preprocess_did_keeps_callable_est_method(mpdta_data):
    dp = preprocess_did(
        mpdta_data, yname="lemp", tname="year", idname="countyreal", gname="first.treat", est_method=drdid_panel
    )

    assert dp.config.est_method is drdid_panel
