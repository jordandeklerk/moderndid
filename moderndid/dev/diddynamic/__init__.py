"""Dynamic covariate balancing for Difference-in-Differences."""

import moderndid.dev.diddynamic.format as _format

from .container import DynBalancingResult
from .dyn_balancing import dyn_balancing

__all__ = [
    "DynBalancingResult",
    "dyn_balancing",
]
