"""Triple difference-in-differences estimators."""

import moderndid.didtriple.format  # noqa: F401
from moderndid.didtriple.agg_ddd import agg_ddd
from moderndid.didtriple.agg_ddd_obj import DDDAggResult
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd, wboot_ddd
from moderndid.didtriple.dgp import gen_dgp_2periods, gen_dgp_mult_periods, generate_simple_ddd_data
from moderndid.didtriple.estimators.ddd_mp import ATTgtResult, DDDMultiPeriodResult, ddd_mp
from moderndid.didtriple.estimators.ddd_panel import DDDPanelResult, ddd_panel

__all__ = [
    "ATTgtResult",
    "DDDAggResult",
    "DDDMultiPeriodResult",
    "DDDPanelResult",
    "agg_ddd",
    "ddd_mp",
    "ddd_panel",
    "gen_dgp_2periods",
    "gen_dgp_mult_periods",
    "generate_simple_ddd_data",
    "mboot_ddd",
    "wboot_ddd",
]
