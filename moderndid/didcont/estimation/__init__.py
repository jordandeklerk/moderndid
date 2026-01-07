"""Preprocessing functions for continuous treatment DiD."""

from moderndid.core.preprocess import (
    choose_knots_quantile as _choose_knots_quantile,
)
from moderndid.core.preprocess import (
    get_first_difference as _get_first_difference,
)
from moderndid.core.preprocess import (
    get_group as _get_group,
)
from moderndid.core.preprocess import (
    make_balanced_panel as _make_balanced_panel,
)
from moderndid.core.preprocess import (
    map_to_idx as _map_to_idx,
)

from .container import (
    AttgtResult,
    DoseResult,
    GroupTimeATTResult,
    PTEAggteResult,
    PteEmpBootResult,
    PTEParams,
    PTEResult,
)
from .estimators import (
    did_attgt,
    pte_attgt,
)
from .process_aggte import (
    aggregate_att_gt,
    overall_weights,
)
from .process_attgt import (
    multiplier_bootstrap,
    process_att_gt,
)
from .process_dose import (
    _summary_dose_result,
    process_dose_gt,
)
from .process_panel import (
    _build_pte_params,
    pte,
    pte_default,
    setup_pte,
    setup_pte_basic,
    setup_pte_cont,
)

__all__ = [
    "AttgtResult",
    "DoseResult",
    "GroupTimeATTResult",
    "PTEAggteResult",
    "PTEParams",
    "PTEResult",
    "PteEmpBootResult",
    "_choose_knots_quantile",
    "_get_first_difference",
    "_get_group",
    "_make_balanced_panel",
    "_map_to_idx",
    "_summary_dose_result",
    "aggregate_att_gt",
    "did_attgt",
    "multiplier_bootstrap",
    "overall_weights",
    "process_att_gt",
    "process_dose_gt",
    "pte",
    "pte_attgt",
    "setup_pte",
    "setup_pte_basic",
    "setup_pte_cont",
    "_build_pte_params",
    "pte_default",
]
