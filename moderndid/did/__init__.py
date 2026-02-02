"""Difference-in-Differences estimators with multiple time periods."""

import importlib
from typing import Any

from moderndid.core.preprocess import DIDData, preprocess_did

from .aggte import aggte
from .aggte_obj import AGGTEResult, format_aggte_result
from .att_gt import att_gt
from .compute_aggte import compute_aggte
from .compute_att_gt import ATTgtResult, ComputeATTgtResult, compute_att_gt
from .mboot import mboot
from .multiperiod_obj import (
    MPPretestResult,
    MPResult,
    format_mp_pretest_result,
    format_mp_result,
    mp,
    mp_pretest,
    summary_mp_pretest,
)

# Lazy loading for plot functions (requires plotnine)
_PLOT_FUNCTIONS = {"plot_event_study", "plot_gt"}


def __getattr__(name: str) -> Any:
    """Lazy loading for plot functions."""
    if name in _PLOT_FUNCTIONS:
        try:
            from moderndid.did.plots import plot_event_study, plot_gt

            return plot_event_study if name == "plot_event_study" else plot_gt
        except ImportError as e:
            raise ImportError(f"'{name}' requires extra dependencies: uv pip install 'moderndid[plots]'") from e
    raise AttributeError(f"module 'moderndid.did' has no attribute '{name}'")


__all__ = [
    "AGGTEResult",
    "ATTgtResult",
    "ComputeATTgtResult",
    "DIDData",
    "MPPretestResult",
    "MPResult",
    "aggte",
    "att_gt",
    "compute_aggte",
    "compute_att_gt",
    "format_aggte_result",
    "format_mp_pretest_result",
    "format_mp_result",
    "mboot",
    "mp",
    "mp_pretest",
    "plot_event_study",
    "plot_gt",
    "preprocess_did",
    "summary_mp_pretest",
]
