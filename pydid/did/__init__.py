"""Difference-in-Differences estimators with multiple time periods."""

from .aggte_obj import AGGTEResult, aggte, format_aggte_result
from .multiperiod_obj import (
    MPPretestResult,
    MPResult,
    format_mp_pretest_result,
    format_mp_result,
    mp,
    mp_pretest,
    summary_mp_pretest,
)
from .preprocess_did import preprocess_did
from .preprocessing.models import DIDData

__all__ = [
    "AGGTEResult",
    "aggte",
    "format_aggte_result",
    "MPResult",
    "mp",
    "format_mp_result",
    "MPPretestResult",
    "mp_pretest",
    "format_mp_pretest_result",
    "summary_mp_pretest",
    "preprocess_did",
    "DIDData",
]
