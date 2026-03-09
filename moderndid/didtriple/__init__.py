"""Triple difference-in-differences estimators."""

import moderndid.didtriple.format
from moderndid.didtriple.agg_ddd import agg_ddd
from moderndid.didtriple.bootstrap.mboot_ddd import mboot_ddd, wboot_ddd
from moderndid.didtriple.container import (
    ATTgtRCResult,
    ATTgtResult,
    DDDAggResult,
    DDDMultiPeriodRCResult,
    DDDMultiPeriodResult,
    DDDPanelResult,
    DDDRCResult,
)
from moderndid.didtriple.ddd import ddd
from moderndid.didtriple.estimators.ddd_mp import ddd_mp
from moderndid.didtriple.estimators.ddd_mp_rc import ddd_mp_rc
from moderndid.didtriple.estimators.ddd_panel import ddd_panel
from moderndid.didtriple.estimators.ddd_rc import ddd_rc

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
    "gen_ddd_2periods",
    "gen_ddd_mult_periods",
    "gen_ddd_scalable",
    "gen_simple_ddd_data",
    "generate_simple_ddd_data",
    "mboot_ddd",
    "wboot_ddd",
]

_DGP_NAMES = {
    "gen_ddd_2periods",
    "gen_ddd_mult_periods",
    "gen_ddd_scalable",
    "gen_simple_ddd_data",
    "generate_simple_ddd_data",
}


def __getattr__(name):
    if name in _DGP_NAMES:
        import moderndid.core.data as _data

        return getattr(_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
