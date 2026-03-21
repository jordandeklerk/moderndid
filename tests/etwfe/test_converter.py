"""Tests for ETWFE result-to-polars conversion."""

import numpy as np
import polars as pl
import pytest

from tests.helpers import importorskip

importorskip("pyfixest")

from moderndid import emfx
from moderndid.core.converters import emfxresult_to_polars


@pytest.mark.parametrize("agg_type", ["event", "group"])
def test_emfxresult_to_polars(etwfe_baseline, agg_type):
    result = emfx(etwfe_baseline, type=agg_type)
    df = emfxresult_to_polars(result)

    assert isinstance(df, pl.DataFrame)
    for col in ("event_time", "att", "se", "ci_lower", "ci_upper"):
        assert col in df.columns
    assert len(df) == len(result.event_times)
    np.testing.assert_allclose(df["att"].to_numpy(), result.att_by_event, atol=1e-10)
    np.testing.assert_array_equal(df["event_time"].to_numpy(), result.event_times)


def test_emfxresult_to_polars_event_has_treatment_status(etwfe_baseline):
    result = emfx(etwfe_baseline, type="event")
    df = emfxresult_to_polars(result)
    assert "treatment_status" in df.columns


def test_emfxresult_to_polars_simple_raises(etwfe_baseline):
    result = emfx(etwfe_baseline, type="simple")
    with pytest.raises(ValueError, match="Simple aggregation"):
        emfxresult_to_polars(result)
