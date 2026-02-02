"""Unified plotting infrastructure for moderndid using plotnine."""

from moderndid.plots.converters import (
    aggteresult_to_polars,
    dddaggresult_to_polars,
    dddmpresult_to_polars,
    didinterresult_to_polars,
    doseresult_to_polars,
    honestdid_to_polars,
    mpresult_to_polars,
    pteresult_to_polars,
)
from moderndid.plots.plots import (
    plot_agg,
    plot_dose_response,
    plot_event_study,
    plot_gt,
    plot_multiplegt,
    plot_sensitivity,
)
from moderndid.plots.themes import (
    COLORS,
    theme_minimal,
    theme_moderndid,
    theme_publication,
)

__all__ = [
    "COLORS",
    "aggteresult_to_polars",
    "dddaggresult_to_polars",
    "dddmpresult_to_polars",
    "didinterresult_to_polars",
    "doseresult_to_polars",
    "honestdid_to_polars",
    # Converters
    "mpresult_to_polars",
    "plot_agg",
    "plot_dose_response",
    "plot_event_study",
    # Plot functions
    "plot_gt",
    "plot_multiplegt",
    "plot_sensitivity",
    "pteresult_to_polars",
    "theme_minimal",
    # Themes
    "theme_moderndid",
    "theme_publication",
]
