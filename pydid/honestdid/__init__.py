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
from .delta_sdrm import (
    DeltaSDRMResult,
    compute_conditional_cs_sdrm,
    compute_identified_set_sdrm,
)
from .delta_sdrmb import (
    DeltaSDRMBResult,
    compute_conditional_cs_sdrmb,
    compute_identified_set_sdrmb,
)
from .delta_sdrmm import (
    DeltaSDRMMResult,
    compute_conditional_cs_sdrmm,
    compute_identified_set_sdrmm,
)
from .did_sunab import (
    SunAbrahamResult,
    aggregate_sunab,
    sunab,
    sunab_att,
)
from .fixed_length_ci import (
    FLCIResult,
    compute_flci,
)
from .honest_did import (
    HonestDiDResult,
    honest_did,
)
from .honest_sunab import (
    SunAbrahamCoefficients,
    extract_sunab_coefficients,
)
from .numba import (
    compute_bounds,
    lee_coefficient,
    selection_matrix,
)
from .plots.core import (
    event_study_plot,
    plot_sensitivity,
    plot_sensitivity_rm,
)
from .sensitivity import (
    OriginalCSResult,
    SensitivityResult,
    construct_original_cs,
    create_sensitivity_results,
    create_sensitivity_results_relative_magnitudes,
)
from .utils import (
    basis_vector,
    bin_factor,
    create_interactions,
    validate_conformable,
    validate_symmetric_psd,
)

__all__ = [
    # Main interface
    "honest_did",
    "HonestDiDResult",
    # Sun & Abraham coefficient extraction
    "extract_sunab_coefficients",
    "SunAbrahamCoefficients",
    # Sun-Abraham estimator
    "sunab",
    "sunab_att",
    "aggregate_sunab",
    "SunAbrahamResult",
    # Utility functions
    "basis_vector",
    "validate_symmetric_psd",
    "validate_conformable",
    "lee_coefficient",
    "bin_factor",
    "create_interactions",
    "selection_matrix",
    "compute_bounds",
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
    # Delta SDRM (second differences with relative magnitudes)
    "DeltaSDRMResult",
    "compute_conditional_cs_sdrm",
    "compute_identified_set_sdrm",
    # Delta SDRMB (second differences with relative magnitudes and bias)
    "DeltaSDRMBResult",
    "compute_conditional_cs_sdrmb",
    "compute_identified_set_sdrmb",
    # Delta SDRMM (second differences with relative magnitudes and monotonicity)
    "DeltaSDRMMResult",
    "compute_conditional_cs_sdrmm",
    "compute_identified_set_sdrmm",
    # Main sensitivity analysis functions
    "OriginalCSResult",
    "SensitivityResult",
    "construct_original_cs",
    "create_sensitivity_results",
    "create_sensitivity_results_relative_magnitudes",
    # Plotting functions
    "event_study_plot",
    "plot_sensitivity",
    "plot_sensitivity_rm",
]
