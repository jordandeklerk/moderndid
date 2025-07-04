# pylint: disable=wildcard-import
"""DiD and doubly robust DiD estimators."""

from pydid.data import load_mpdta, load_nsw
from pydid.did.aggte import aggte
from pydid.did.aggte_obj import AGGTEResult, format_aggte_result
from pydid.did.att_gt import att_gt
from pydid.did.compute_aggte import compute_aggte
from pydid.did.compute_att_gt import ATTgtResult, ComputeATTgtResult, compute_att_gt
from pydid.did.mboot import mboot
from pydid.did.multiperiod_obj import (
    MPPretestResult,
    MPResult,
    format_mp_pretest_result,
    format_mp_result,
    mp,
    mp_pretest,
    summary_mp_pretest,
)
from pydid.did.plots import (
    plot_att_gt,
    plot_did,
    plot_event_study,
)
from pydid.did.preprocess import DIDData
from pydid.did.preprocess_did import preprocess_did
from pydid.drdid.bootstrap.boot_ipw_rc import wboot_ipw_rc
from pydid.drdid.bootstrap.boot_mult import mboot_did, mboot_twfep_did
from pydid.drdid.bootstrap.boot_panel import (
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_ipw_panel,
    wboot_reg_panel,
    wboot_std_ipw_panel,
    wboot_twfe_panel,
)
from pydid.drdid.bootstrap.boot_rc import wboot_drdid_rc1, wboot_drdid_rc2
from pydid.drdid.bootstrap.boot_rc_ipt import wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2
from pydid.drdid.bootstrap.boot_reg_rc import wboot_reg_rc
from pydid.drdid.bootstrap.boot_std_ipw_rc import wboot_std_ipw_rc
from pydid.drdid.bootstrap.boot_twfe_rc import wboot_twfe_rc
from pydid.drdid.drdid import drdid
from pydid.drdid.estimators.drdid_imp_local_rc import drdid_imp_local_rc
from pydid.drdid.estimators.drdid_imp_panel import drdid_imp_panel
from pydid.drdid.estimators.drdid_imp_rc import drdid_imp_rc
from pydid.drdid.estimators.drdid_panel import drdid_panel
from pydid.drdid.estimators.drdid_rc import drdid_rc
from pydid.drdid.estimators.drdid_trad_rc import drdid_trad_rc
from pydid.drdid.estimators.ipw_did_panel import ipw_did_panel
from pydid.drdid.estimators.ipw_did_rc import ipw_did_rc
from pydid.drdid.estimators.reg_did_panel import reg_did_panel
from pydid.drdid.estimators.reg_did_rc import reg_did_rc
from pydid.drdid.estimators.std_ipw_did_panel import std_ipw_did_panel
from pydid.drdid.estimators.std_ipw_did_rc import std_ipw_did_rc
from pydid.drdid.estimators.twfe_did_panel import twfe_did_panel
from pydid.drdid.estimators.twfe_did_rc import twfe_did_rc
from pydid.drdid.estimators.wols import wols_panel, wols_rc
from pydid.drdid.ipwdid import ipwdid
from pydid.drdid.ordid import ordid
from pydid.drdid.print import print_did_result
from pydid.drdid.propensity.aipw_estimators import aipw_did_panel, aipw_did_rc_imp1, aipw_did_rc_imp2
from pydid.drdid.propensity.ipw_estimators import ipw_rc
from pydid.drdid.propensity.pscore_ipt import calculate_pscore_ipt
from pydid.honestdid import (
    APRCIResult,
    ARPNuisanceCIResult,
    DeltaRMBResult,
    DeltaRMMResult,
    DeltaRMResult,
    DeltaSDBResult,
    DeltaSDMResult,
    DeltaSDResult,
    DeltaSDRMResult,
    FLCIResult,
    basis_vector,
    compute_arp_ci,
    compute_arp_nuisance_ci,
    compute_bounds,
    compute_conditional_cs_rm,
    compute_conditional_cs_rmb,
    compute_conditional_cs_rmm,
    compute_conditional_cs_sd,
    compute_conditional_cs_sdb,
    compute_conditional_cs_sdm,
    compute_conditional_cs_sdrm,
    compute_delta_sd_lowerbound_m,
    compute_delta_sd_upperbound_m,
    compute_flci,
    compute_identified_set_rm,
    compute_identified_set_rmb,
    compute_identified_set_rmm,
    compute_identified_set_sd,
    compute_identified_set_sdb,
    compute_identified_set_sdm,
    compute_identified_set_sdrm,
    create_monotonicity_constraint_matrix,
    create_pre_period_constraint_matrix,
    create_second_difference_matrix,
    create_sign_constraint_matrix,
    estimate_lowerbound_m_conditional_test,
    lee_coefficient,
    selection_matrix,
    test_in_identified_set_max,
    validate_conformable,
    validate_symmetric_psd,
)
from pydid.utils import (
    are_varying,
    complete_data,
    convert_panel_time_to_int,
    create_relative_time_indicators,
    datetime_to_int,
    extract_vars_from_formula,
    fill_panel_gaps,
    is_panel_balanced,
    is_repeated_cross_section,
    long_panel,
    make_panel_balanced,
    panel_has_gaps,
    panel_to_cross_section_diff,
    parse_formula,
    prepare_data_for_did,
    unpanel,
    validate_treatment_timing,
    widen_panel,
)

__all__ = [
    # DR-DiD estimators
    "drdid",
    "drdid_imp_panel",
    "drdid_imp_rc",
    "drdid_imp_local_rc",
    "drdid_panel",
    "drdid_rc",
    "drdid_trad_rc",
    # IPW DiD estimators
    "ipwdid",
    "ipw_did_panel",
    "ipw_did_rc",
    "std_ipw_did_panel",
    "std_ipw_did_rc",
    # Outcome regression estimators
    "ordid",
    "reg_did_panel",
    "reg_did_rc",
    "twfe_did_panel",
    "twfe_did_rc",
    # Core propensity score estimators
    "aipw_did_panel",
    "aipw_did_rc_imp1",
    "aipw_did_rc_imp2",
    "ipw_rc",
    "calculate_pscore_ipt",
    # Bootstrap functions
    "mboot",
    "mboot_did",
    "mboot_twfep_did",
    "wboot_dr_tr_panel",
    "wboot_drdid_imp_panel",
    "wboot_drdid_rc1",
    "wboot_drdid_rc2",
    "wboot_drdid_ipt_rc1",
    "wboot_drdid_ipt_rc2",
    "wboot_ipw_panel",
    "wboot_ipw_rc",
    "wboot_reg_panel",
    "wboot_reg_rc",
    "wboot_std_ipw_panel",
    "wboot_std_ipw_rc",
    "wboot_twfe_panel",
    "wboot_twfe_rc",
    # Regression functions
    "wols_panel",
    "wols_rc",
    # Panel data utilities
    "are_varying",
    "complete_data",
    "convert_panel_time_to_int",
    "create_relative_time_indicators",
    "datetime_to_int",
    "extract_vars_from_formula",
    "fill_panel_gaps",
    "is_panel_balanced",
    "is_repeated_cross_section",
    "long_panel",
    "make_panel_balanced",
    "panel_has_gaps",
    "panel_to_cross_section_diff",
    "parse_formula",
    "prepare_data_for_did",
    "unpanel",
    "validate_treatment_timing",
    "widen_panel",
    # Print function
    "print_did_result",
    # Datasets module
    "load_nsw",
    "load_mpdta",
    # Multi-period result objects
    "MPResult",
    "mp",
    "format_mp_result",
    "MPPretestResult",
    "mp_pretest",
    "format_mp_pretest_result",
    "summary_mp_pretest",
    # Aggregate treatment effect result objects
    "AGGTEResult",
    "aggte",
    "format_aggte_result",
    # Preprocessing functions
    "DIDData",
    "preprocess_did",
    # Multi-period DiD computation
    "att_gt",
    "ATTgtResult",
    "ComputeATTgtResult",
    "compute_att_gt",
    "compute_aggte",
    # Honest DiD utility functions
    "selection_matrix",
    "compute_bounds",
    "basis_vector",
    "validate_symmetric_psd",
    "validate_conformable",
    "lee_coefficient",
    # Honest DiD bound estimation
    "compute_delta_sd_upperbound_m",
    "compute_delta_sd_lowerbound_m",
    "create_second_difference_matrix",
    "create_pre_period_constraint_matrix",
    "create_monotonicity_constraint_matrix",
    "create_sign_constraint_matrix",
    # Honest DiD conditional test
    "test_in_identified_set_max",
    "estimate_lowerbound_m_conditional_test",
    # Honest DiD FLCI
    "compute_flci",
    "FLCIResult",
    # Honest DiD APR CI
    "compute_arp_ci",
    "APRCIResult",
    "compute_arp_nuisance_ci",
    "ARPNuisanceCIResult",
    # Honest DiD Delta RM
    "DeltaRMResult",
    "compute_conditional_cs_rm",
    "compute_identified_set_rm",
    # Honest DiD Delta RMB
    "DeltaRMBResult",
    "compute_conditional_cs_rmb",
    "compute_identified_set_rmb",
    # Honest DiD Delta RMM
    "DeltaRMMResult",
    "compute_conditional_cs_rmm",
    "compute_identified_set_rmm",
    # Honest DiD Delta SD
    "DeltaSDResult",
    "compute_conditional_cs_sd",
    "compute_identified_set_sd",
    # Honest DiD Delta SDB
    "DeltaSDBResult",
    "compute_conditional_cs_sdb",
    "compute_identified_set_sdb",
    # Honest DiD Delta SDM
    "DeltaSDMResult",
    "compute_conditional_cs_sdm",
    "compute_identified_set_sdm",
    # Honest DiD Delta SDRM
    "DeltaSDRMResult",
    "compute_conditional_cs_sdrm",
    "compute_identified_set_sdrm",
    # Plotting functions
    "plot_att_gt",
    "plot_event_study",
    "plot_did",
]
