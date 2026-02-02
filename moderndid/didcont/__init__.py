"""Difference-in-differences with a continuous treatment."""

from moderndid.core.preprocess import (
    get_first_difference as _get_first_difference,
)
from moderndid.core.preprocess import (
    get_group as _get_group,
)
from moderndid.core.preprocess import (
    make_balanced_panel as _make_balanced_panel,
)

from .cont_did import cont_did
from .estimation import (
    DoseResult,
    GroupTimeATTResult,
    PTEAggteResult,
    PTEParams,
    _summary_dose_result,
    aggregate_att_gt,
    did_attgt,
    multiplier_bootstrap,
    overall_weights,
    process_att_gt,
    process_dose_gt,
    pte_attgt,
    setup_pte,
    setup_pte_basic,
    setup_pte_cont,
)
from .npiv import (
    BSplineBasis,
    MultivariateBasis,
    NPIVResult,
    compute_ucb,
    gsl_bs,
    npiv,
    npiv_choose_j,
    npiv_est,
    predict_gsl_bs,
    prodspline,
)
from .utils import (
    _quantile_basis,
    avoid_zero_division,
    basis_dimension,
    is_full_rank,
    matrix_sqrt,
)

__all__ = [
    # B-spline and basis construction
    "BSplineBasis",
    "DoseResult",
    "GroupTimeATTResult",
    "MultivariateBasis",
    # Result types
    "NPIVResult",
    "PTEAggteResult",
    # Panel treatment effects setup
    "PTEParams",
    "_get_first_difference",
    "_get_group",
    "_make_balanced_panel",
    "_quantile_basis",
    "_summary_dose_result",
    "aggregate_att_gt",
    "avoid_zero_division",
    "basis_dimension",
    "compute_ucb",
    "cont_did",
    "did_attgt",
    "gsl_bs",
    # Utility functions
    "is_full_rank",
    "matrix_sqrt",
    "multiplier_bootstrap",
    # Main NPIV estimation functions
    "npiv",
    # Dimension selection functions
    "npiv_choose_j",
    "npiv_est",
    "overall_weights",
    "predict_gsl_bs",
    # Processing functions
    "process_att_gt",
    "process_dose_gt",
    "prodspline",
    "pte_attgt",
    "setup_pte",
    "setup_pte_basic",
    "setup_pte_cont",
]
