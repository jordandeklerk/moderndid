"""Preprocessing functions for continuous treatment DiD."""

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
    PTEParams,
    _choose_knots_quantile,
    _get_first_difference,
    _get_group,
    _get_group_inner,
    _make_balanced_panel,
    _map_to_idx,
    setup_pte,
    setup_pte_basic,
    setup_pte_cont,
)

__all__ = [
    # Setup functions
    "PTEParams",
    "setup_pte",
    "setup_pte_basic",
    "setup_pte_cont",
    # Processing functions
    "process_att_gt",
    "GroupTimeATTResult",
    "aggregate_att_gt",
    "PTEAggteResult",
    "process_dose_gt",
    "DoseResult",
    "_summary_dose_result",
    "overall_weights",
    "multiplier_bootstrap",
    # Helper functions
    "_make_balanced_panel",
    "_get_first_difference",
    "_get_group",
    "_get_group_inner",
    "_map_to_idx",
    "_choose_knots_quantile",
]
