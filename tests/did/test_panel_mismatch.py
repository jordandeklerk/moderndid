"""Tests for panel/RCS mismatch detection."""

import re

import polars as pl
import pytest

from moderndid.core.preprocess.config import DDDConfig, DIDConfig
from moderndid.core.preprocess.validators import (
    DDDPanelStructureValidator,
    _check_panel_mismatch,
)
from moderndid.didtriple.utils import detect_rcs_mode


@pytest.mark.parametrize("idname", [None, ""])
def test_check_panel_mismatch_no_idname(panel_df, idname):
    errors, warnings = _check_panel_mismatch(panel_df, idname, "time", panel=True)

    assert errors == []
    assert warnings == []


def test_check_panel_mismatch_panel_true_with_rcs(rcs_df):
    errors, warnings = _check_panel_mismatch(rcs_df, "id", "time", panel=True)

    assert errors == [
        "panel=True was specified, but no units appear in multiple time periods. "
        "Your data appears to be repeated cross-sections. "
        "Set panel=False to use the repeated cross-section estimator."
    ]
    assert warnings == []


def test_check_panel_mismatch_panel_false_with_panel(panel_df):
    errors, warnings = _check_panel_mismatch(panel_df, "id", "time", panel=False)

    assert errors == []
    assert warnings == [
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator."
    ]


def test_check_panel_mismatch_correct_panel_true(panel_df):
    errors, warnings = _check_panel_mismatch(panel_df, "id", "time", panel=True)

    assert errors == []
    assert warnings == []


def test_check_panel_mismatch_correct_panel_false(rcs_df):
    errors, warnings = _check_panel_mismatch(rcs_df, "id", "time", panel=False)

    assert errors == []
    assert warnings == []


@pytest.mark.parametrize(
    ("panel", "expected"),
    [
        (True, []),
        (
            False,
            [
                "panel=False was specified, but units appear across all time periods. "
                "Your data appears to be panel data. Consider setting panel=True to use the panel estimator."
            ],
        ),
    ],
)
def test_check_panel_mismatch_reads_an_unbalanced_panel_as_a_panel(panel_df, panel, expected):
    unbalanced = panel_df.filter((pl.col("id") != 1) | (pl.col("time") == 1))

    errors, warnings = _check_panel_mismatch(unbalanced, "id", "time", panel=panel)

    assert errors == []
    assert warnings == expected


def test_check_panel_mismatch_single_period():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "time": [1, 1, 1],
            "y": [1.0, 2.0, 3.0],
        }
    )

    errors, warnings = _check_panel_mismatch(df, "id", "time", panel=False)

    assert errors == []
    assert warnings == []


def test_ddd_panel_structure_validator_panel_true_with_rcs(rcs_df):
    rcs_df = rcs_df.with_columns(pl.lit(1).alias("p"))
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p", panel=True)

    result = DDDPanelStructureValidator().validate(rcs_df, config)

    assert not result.is_valid
    assert result.errors == [
        "panel=True was specified, but no units appear in multiple time periods. "
        "Your data appears to be repeated cross-sections. "
        "Set panel=False to use the repeated cross-section estimator."
    ]
    assert result.warnings == []


def test_ddd_panel_structure_validator_panel_false_with_panel(panel_df):
    panel_df = panel_df.with_columns(pl.lit(1).alias("p"))
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p", panel=False)

    result = DDDPanelStructureValidator().validate(panel_df, config)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == [
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator."
    ]


def test_ddd_panel_structure_validator_correct(panel_df):
    panel_df = panel_df.with_columns(pl.lit(1).alias("p"))
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p", panel=True)

    result = DDDPanelStructureValidator().validate(panel_df, config)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


@pytest.mark.parametrize(
    ("panel", "expected"),
    [
        (
            True,
            ["The value of idname must be unique (by tname). Some units are observed more than once in a period."],
        ),
        (False, []),
    ],
)
def test_ddd_panel_structure_validator_flags_repeated_unit_periods_only_in_a_panel(panel_df, panel, expected):
    repeated = pl.concat([panel_df, panel_df.head(1)]).with_columns(pl.lit(1).alias("p"))
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p", panel=panel)

    result = DDDPanelStructureValidator().validate(repeated, config)

    assert result.errors == expected


def test_ddd_panel_structure_validator_skips_non_ddd_config(rcs_df):
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g", panel=True)

    result = DDDPanelStructureValidator().validate(rcs_df, config)

    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_detect_rcs_mode_panel_true_with_rcs_raises(rcs_df):
    message = (
        "panel=True was specified, but no units appear in multiple time periods. "
        "Your data appears to be repeated cross-sections. "
        "Set panel=False to use the repeated cross-section estimator."
    )

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        detect_rcs_mode(rcs_df, "time", "id", panel=True, allow_unbalanced_panel=False)


def test_detect_rcs_mode_panel_false_with_panel_warns(panel_df):
    message = (
        "panel=False was specified, but units appear across all time periods. Your data appears to be panel data. "
        "Consider setting panel=True to use the panel estimator."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(message)}$") as record:
        result = detect_rcs_mode(panel_df, "time", "id", panel=False, allow_unbalanced_panel=False)

    assert result is True
    assert len(record) == 1


def test_ddd_config_panel_defaults():
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p")

    assert config.panel is True
    assert config.allow_unbalanced_panel is False


@pytest.mark.parametrize("panel", [True, False])
def test_ddd_config_panel_roundtrip(panel):
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p", panel=panel)

    d = config.to_dict()

    assert d["panel"] is panel
