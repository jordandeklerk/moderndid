"""Difference-in-Differences estimators with multiple time periods."""

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
from .plots import (
    plot_att_gt,
    plot_event_study,
)

__all__ = [
    "aggte",
    "AGGTEResult",
    "format_aggte_result",
    "compute_aggte",
    "att_gt",
    "MPResult",
    "mp",
    "format_mp_result",
    "MPPretestResult",
    "mp_pretest",
    "format_mp_pretest_result",
    "summary_mp_pretest",
    "preprocess_did",
    "DIDData",
    "mboot",
    "ATTgtResult",
    "ComputeATTgtResult",
    "compute_att_gt",
    "plot_att_gt",
    "plot_event_study",
]
