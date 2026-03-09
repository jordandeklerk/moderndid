"""Core DDD estimators."""

from moderndid.didtriple.container import (
    ATTgtRCResult,
    ATTgtResult,
    DDDMultiPeriodRCResult,
    DDDMultiPeriodResult,
    DDDPanelResult,
    DDDRCResult,
)
from moderndid.didtriple.estimators.ddd_mp import ddd_mp
from moderndid.didtriple.estimators.ddd_mp_rc import ddd_mp_rc
from moderndid.didtriple.estimators.ddd_panel import ddd_panel
from moderndid.didtriple.estimators.ddd_rc import ddd_rc

__all__ = [
    "ATTgtRCResult",
    "ATTgtResult",
    "DDDMultiPeriodRCResult",
    "DDDMultiPeriodResult",
    "DDDPanelResult",
    "DDDRCResult",
    "ddd_mp",
    "ddd_mp_rc",
    "ddd_panel",
    "ddd_rc",
]
