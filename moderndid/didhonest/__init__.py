"""Sensitivity analysis for difference-in-differences using the approach of Rambachan and Roth (2022)."""

try:
    import cvxpy
    import sympy
except ImportError as e:
    raise ImportError(
        "The 'didhonest' module requires additional dependencies. "
        "Install them with: uv pip install 'moderndid[didhonest]'"
    ) from e

from .arp_no_nuisance import (
    APRCIResult,
    compute_arp_ci,
    test_in_identified_set,
    test_in_identified_set_flci_hybrid,
    test_in_identified_set_lf_hybrid,
)
from .arp_nuisance import (
    ARPNuisanceCIResult,
    compute_arp_nuisance_ci,
    compute_least_favorable_cv,
    compute_vlo_vup_dual,
    lp_conditional_test,
)
from .bounds import (
    compute_delta_sd_lowerbound_m,
    compute_delta_sd_upperbound_m,
    create_monotonicity_constraint_matrix,
    create_pre_period_constraint_matrix,
    create_second_difference_matrix,
    create_sign_constraint_matrix,
)
from .conditional import (
    estimate_lowerbound_m_conditional_test,
    test_in_identified_set_max,
)
from .delta.rm.rm import (
    DeltaRMResult,
    compute_conditional_cs_rm,
    compute_identified_set_rm,
)
from .delta.rm.rmb import (
    DeltaRMBResult,
    compute_conditional_cs_rmb,
    compute_identified_set_rmb,
)
from .delta.rm.rmm import (
    DeltaRMMResult,
    compute_conditional_cs_rmm,
    compute_identified_set_rmm,
)
from .delta.sd.sd import (
    DeltaSDResult,
    compute_conditional_cs_sd,
    compute_identified_set_sd,
)
from .delta.sd.sdb import (
    DeltaSDBResult,
    compute_conditional_cs_sdb,
    compute_identified_set_sdb,
)
from .delta.sd.sdm import (
    DeltaSDMResult,
    compute_conditional_cs_sdm,
    compute_identified_set_sdm,
)
from .delta.sdrm.sdrm import (
    DeltaSDRMResult,
    compute_conditional_cs_sdrm,
    compute_identified_set_sdrm,
)
from .delta.sdrm.sdrmb import (
    DeltaSDRMBResult,
    compute_conditional_cs_sdrmb,
    compute_identified_set_sdrmb,
)
from .delta.sdrm.sdrmm import (
    DeltaSDRMMResult,
    compute_conditional_cs_sdrmm,
    compute_identified_set_sdrmm,
)
from .fixed_length_ci import (
    FLCIResult,
    affine_variance,
    compute_flci,
    folded_normal_quantile,
    maximize_bias,
    minimize_variance,
)
from .honest_did import (
    HonestDiDResult,
    honest_did,
)
from .numba import (
    compute_bounds,
    lee_coefficient,
    selection_matrix,
)
from .sensitivity import (
    OriginalCSResult,
    SensitivityResult,
    construct_original_cs,
    create_sensitivity_results_rm,
    create_sensitivity_results_sm,
)
from .utils import (
    basis_vector,
    bin_factor,
    create_interactions,
    validate_conformable,
    validate_symmetric_psd,
)
from .wrappers import (
    DeltaMethodSelector,
    get_delta_method,
)

__all__ = [
    "APRCIResult",
    "ARPNuisanceCIResult",
    # Wrapper utilities
    "DeltaMethodSelector",
    # Delta RMB (relative magnitudes with bias restriction)
    "DeltaRMBResult",
    # Delta RMM (relative magnitudes with monotonicity restriction)
    "DeltaRMMResult",
    # Delta RM (relative magnitudes)
    "DeltaRMResult",
    # Delta SDB (second differences with bias)
    "DeltaSDBResult",
    # Delta SDM (second differences with monotonicity)
    "DeltaSDMResult",
    # Delta SDRMB (second differences with relative magnitudes and bias)
    "DeltaSDRMBResult",
    # Delta SDRMM (second differences with relative magnitudes and monotonicity)
    "DeltaSDRMMResult",
    # Delta SDRM (second differences with relative magnitudes)
    "DeltaSDRMResult",
    # Delta SD (second differences)
    "DeltaSDResult",
    "FLCIResult",
    "HonestDiDResult",
    # Main sensitivity analysis functions
    "OriginalCSResult",
    "SensitivityResult",
    "affine_variance",
    # Utility functions
    "basis_vector",
    "bin_factor",
    # APR confidence intervals (no nuisance)
    "compute_arp_ci",
    # APR confidence intervals (with nuisance)
    "compute_arp_nuisance_ci",
    "compute_bounds",
    "compute_conditional_cs_rm",
    "compute_conditional_cs_rmb",
    "compute_conditional_cs_rmm",
    "compute_conditional_cs_sd",
    "compute_conditional_cs_sdb",
    "compute_conditional_cs_sdm",
    "compute_conditional_cs_sdrm",
    "compute_conditional_cs_sdrmb",
    "compute_conditional_cs_sdrmm",
    "compute_delta_sd_lowerbound_m",
    # Delta SD bounds
    "compute_delta_sd_upperbound_m",
    # Fixed-length confidence intervals (FLCI)
    "compute_flci",
    "compute_identified_set_rm",
    "compute_identified_set_rmb",
    "compute_identified_set_rmm",
    "compute_identified_set_sd",
    "compute_identified_set_sdb",
    "compute_identified_set_sdm",
    "compute_identified_set_sdrm",
    "compute_identified_set_sdrmb",
    "compute_identified_set_sdrmm",
    "compute_least_favorable_cv",
    "compute_vlo_vup_dual",
    "construct_original_cs",
    "create_interactions",
    "create_monotonicity_constraint_matrix",
    "create_pre_period_constraint_matrix",
    "create_second_difference_matrix",
    "create_sensitivity_results_rm",
    "create_sensitivity_results_sm",
    "create_sign_constraint_matrix",
    "estimate_lowerbound_m_conditional_test",
    "folded_normal_quantile",
    "get_delta_method",
    # Main interface
    "honest_did",
    "lee_coefficient",
    "lp_conditional_test",
    "maximize_bias",
    "minimize_variance",
    "selection_matrix",
    "test_in_identified_set",
    "test_in_identified_set_flci_hybrid",
    "test_in_identified_set_lf_hybrid",
    # Conditional test functions
    "test_in_identified_set_max",
    "validate_conformable",
    "validate_symmetric_psd",
]
