"""Tests for panel/RCS mismatch detection."""

import polars as pl
import pytest

from moderndid.core.preprocess.config import DDDConfig, DIDConfig
from moderndid.core.preprocess.validators import (
    DDDPanelStructureValidator,
    _check_panel_mismatch,
)
from moderndid.didtriple.utils import detect_rcs_mode


@pytest.fixture
def panel_df():
    return pl.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3],
            "time": [1, 2, 1, 2, 1, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": [0, 0, 3, 3, 3, 3],
        }
    )


@pytest.fixture
def rcs_df():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "time": [1, 1, 1, 2, 2, 2],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": [0, 3, 3, 0, 3, 3],
        }
    )


@pytest.mark.parametrize("idname", [None, ""])
def test_check_panel_mismatch_no_idname(panel_df, idname):
    errors, warnings = _check_panel_mismatch(panel_df, idname, "time", panel=True)
    assert errors == []
    assert warnings == []


def test_check_panel_mismatch_panel_true_with_rcs(rcs_df):
    errors, warnings = _check_panel_mismatch(rcs_df, "id", "time", panel=True)
    assert len(errors) == 1
    assert "panel=True was specified" in errors[0]
    assert warnings == []


def test_check_panel_mismatch_panel_false_with_panel(panel_df):
    errors, warnings = _check_panel_mismatch(panel_df, "id", "time", panel=False)
    assert errors == []
    assert len(warnings) == 1
    assert "panel=False was specified" in warnings[0]


def test_check_panel_mismatch_correct_panel_true(panel_df):
    errors, warnings = _check_panel_mismatch(panel_df, "id", "time", panel=True)
    assert errors == []
    assert warnings == []


def test_check_panel_mismatch_correct_panel_false(rcs_df):
    errors, warnings = _check_panel_mismatch(rcs_df, "id", "time", panel=False)
    assert errors == []
    assert warnings == []


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
    validator = DDDPanelStructureValidator()
    config = DDDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        pname="p",
        panel=True,
    )
    result = validator.validate(rcs_df, config)
    assert not result.is_valid
    assert any("panel=True was specified" in err for err in result.errors)


def test_ddd_panel_structure_validator_panel_false_with_panel(panel_df):
    panel_df = panel_df.with_columns(pl.lit(1).alias("p"))
    validator = DDDPanelStructureValidator()
    config = DDDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        pname="p",
        panel=False,
    )
    result = validator.validate(panel_df, config)
    assert result.is_valid
    assert any("panel=False was specified" in w for w in result.warnings)


def test_ddd_panel_structure_validator_correct(panel_df):
    panel_df = panel_df.with_columns(pl.lit(1).alias("p"))
    validator = DDDPanelStructureValidator()
    config = DDDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        pname="p",
        panel=True,
    )
    result = validator.validate(panel_df, config)
    assert result.is_valid
    assert result.errors == []


def test_ddd_panel_structure_validator_skips_non_ddd_config(panel_df):
    validator = DDDPanelStructureValidator()
    config = DIDConfig(yname="y", tname="time", idname="id", gname="g")
    result = validator.validate(panel_df, config)
    assert result.is_valid


def test_detect_rcs_mode_panel_true_with_rcs_raises(rcs_df):
    with pytest.raises(ValueError, match="panel=True was specified"):
        detect_rcs_mode(rcs_df, "time", "id", panel=True, allow_unbalanced_panel=False)


def test_detect_rcs_mode_panel_false_with_panel_warns(panel_df):
    import warnings as w

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        result = detect_rcs_mode(panel_df, "time", "id", panel=False, allow_unbalanced_panel=False)
    assert result is True
    assert len(caught) == 1
    assert "panel=False was specified" in str(caught[0].message)


def test_ddd_config_panel_defaults():
    config = DDDConfig(yname="y", tname="time", idname="id", gname="g", pname="p")
    assert config.panel is True
    assert config.allow_unbalanced_panel is False


@pytest.mark.parametrize("panel", [True, False])
def test_ddd_config_panel_roundtrip(panel):
    config = DDDConfig(
        yname="y",
        tname="time",
        idname="id",
        gname="g",
        pname="p",
        panel=panel,
    )
    d = config.to_dict()
    assert d["panel"] is panel
