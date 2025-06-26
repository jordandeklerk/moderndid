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
    create_pre_period_constraint_matrix,
    create_second_difference_matrix,
)
from .conditional import (
    estimate_lowerbound_m_conditional_test,
    test_in_identified_set_max,
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
]
