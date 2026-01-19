"""Triple difference-in-differences estimators."""

import moderndid.didtriple.format  # noqa: F401
from moderndid.didtriple.agg_ddd import agg_ddd
from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd, wboot_ddd
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods, generate_simple_ddd_data
from moderndid.didtriple.estimators.ddd_mp import ATTgtResult, DDDMultiPeriodResult, ddd_mp
from moderndid.didtriple.estimators.ddd_mp_rc import ATTgtRCResult, DDDMultiPeriodRCResult, ddd_mp_rc
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult, ddd_panel
from moderndid.didtriple.estimators.ddd_rc import DDDRCResult, ddd_rc

__all__ = [
    "ATTgtRCResult",
    "ATTgtResult",
    "DDDAggResult",
    "DDDMultiPeriodRCResult",
    "DDDMultiPeriodResult",
    "DDDPanelResult",
    "DDDRCResult",
    "agg_ddd",
    "ddd",
    "ddd_mp",
    "ddd_mp_rc",
    "ddd_panel",
    "ddd_rc",
    "gen_dgp_2periods",
    "gen_dgp_mult_periods",
    "generate_simple_ddd_data",
    "mboot_ddd",
    "wboot_ddd",
]
