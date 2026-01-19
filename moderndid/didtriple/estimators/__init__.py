"""Core DDD estimators."""

from moderndid.didtriple.estimators.ddd_mp import ATTgtResult, DDDMultiPeriodResult, ddd_mp
from moderndid.didtriple.estimators.ddd_mp_rc import ATTgtRCResult, DDDMultiPeriodRCResult, ddd_mp_rc
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult, ddd_panel
from moderndid.didtriple.estimators.ddd_rc import DDDRCResult, ddd_rc

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
