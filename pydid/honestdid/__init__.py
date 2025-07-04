"""Sensitivity analysis for difference-in-differences using the approach of Rambachan and Roth (2022)."""

from .arp_no_nuisance import (
    APRCIResult,
    compute_arp_ci,
)
from .arp_nuisance import (
    ARPNuisanceCIResult,
    compute_arp_nuisance_ci,
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
from .delta_rm import (
    DeltaRMResult,
    compute_conditional_cs_rm,
    compute_identified_set_rm,
)
from .delta_rmb import (
    DeltaRMBResult,
    compute_conditional_cs_rmb,
    compute_identified_set_rmb,
)
from .delta_rmm import (
    DeltaRMMResult,
    compute_conditional_cs_rmm,
    compute_identified_set_rmm,
)
from .delta_sd import (
    DeltaSDResult,
    compute_conditional_cs_sd,
    compute_identified_set_sd,
)
from .delta_sdb import (
    DeltaSDBResult,
    compute_conditional_cs_sdb,
    compute_identified_set_sdb,
)
from .delta_sdm import (
    DeltaSDMResult,
    compute_conditional_cs_sdm,
    compute_identified_set_sdm,
)
from .fixed_length_ci import (
    FLCIResult,
    compute_flci,
)
from .utils import (
    basis_vector,
    compute_bounds,
    lee_coefficient,
    selection_matrix,
    validate_conformable,
    validate_symmetric_psd,
)

__all__ = [
    # Utility functions
    "selection_matrix",
    "lee_coefficient",
    "compute_bounds",
    "basis_vector",
    "validate_symmetric_psd",
    "validate_conformable",
    # Delta SD bounds
    "compute_delta_sd_upperbound_m",
    "compute_delta_sd_lowerbound_m",
    "create_second_difference_matrix",
    "create_pre_period_constraint_matrix",
    "create_monotonicity_constraint_matrix",
    "create_sign_constraint_matrix",
    # Conditional test functions
    "test_in_identified_set_max",
    "estimate_lowerbound_m_conditional_test",
    # Fixed-length confidence intervals (FLCI)
    "compute_flci",
    "FLCIResult",
    # APR confidence intervals (no nuisance)
    "compute_arp_ci",
    "APRCIResult",
    # APR confidence intervals (with nuisance)
    "compute_arp_nuisance_ci",
    "ARPNuisanceCIResult",
    # Delta RM (relative magnitudes)
    "DeltaRMResult",
    "compute_identified_set_rm",
    "compute_conditional_cs_rm",
    # Delta RMB (relative magnitudes with bias restriction)
    "DeltaRMBResult",
    "compute_conditional_cs_rmb",
    "compute_identified_set_rmb",
    # Delta RMM (relative magnitudes with monotonicity restriction)
    "DeltaRMMResult",
    "compute_conditional_cs_rmm",
    "compute_identified_set_rmm",
    # Delta SD (second differences)
    "DeltaSDResult",
    "compute_conditional_cs_sd",
    "compute_identified_set_sd",
    # Delta SDB (second differences with bias)
    "DeltaSDBResult",
    "compute_conditional_cs_sdb",
    "compute_identified_set_sdb",
    # Delta SDM (second differences with monotonicity)
    "DeltaSDMResult",
    "compute_conditional_cs_sdm",
    "compute_identified_set_sdm",
]
