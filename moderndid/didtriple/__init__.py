"""Triple difference-in-differences estimators."""

from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd, wboot_ddd
from moderndid.didtriple.dgp import gen_dgp_2periods
from moderndid.didtriple.estimators.ddd_mp import ATTgtResult, DDDMultiPeriodResult, ddd_mp
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult, ddd_panel

__all__ = [
    "ATTgtResult",
    "DDDMultiPeriodResult",
    "DDDPanelResult",
    "ddd_mp",
    "ddd_panel",
    "gen_dgp_2periods",
    "mboot_ddd",
    "wboot_ddd",
]
