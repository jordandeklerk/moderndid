"""Preprocessing functions for continuous treatment DiD."""

from .estimators import (
    did_attgt,
    pte_attgt,
)
from .process_aggte import (
    PTEAggteResult,
    aggregate_att_gt,
    overall_weights,
)
from .process_attgt import (
    GroupTimeATTResult,
    multiplier_bootstrap,
    process_att_gt,
)
from .process_dose import (
    DoseResult,
    _summary_dose_result,
    process_dose_gt,
)
from .process_panel import (
    AttgtResult,
    PTEParams,
    PTEResult,
    _choose_knots_quantile,
    _get_first_difference,
    _get_group,
    _get_group_inner,
    _make_balanced_panel,
    _map_to_idx,
    pte,
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
    "_choose_knots_quantile",
    "_get_first_difference",
    "_get_group",
    "_get_group_inner",
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
]
