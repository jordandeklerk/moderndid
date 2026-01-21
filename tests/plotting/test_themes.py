"""Tests for plotnine themes."""

import pytest

plotnine = pytest.importorskip("plotnine")

from plotnine import theme

from moderndid.plots.themes import (
    COLORS,
    theme_minimal,
    theme_moderndid,
    theme_publication,
)


def test_colors_dict_has_required_keys():
    assert "pre_treatment" in COLORS
    assert "post_treatment" in COLORS
    assert "line" in COLORS
    assert "ci_fill" in COLORS


def test_theme_moderndid_returns_theme():
    t = theme_moderndid()
    assert isinstance(t, theme)


def test_theme_publication_returns_theme():
    t = theme_publication()
    assert isinstance(t, theme)


def test_theme_minimal_returns_theme():
    t = theme_minimal()
    assert isinstance(t, theme)
