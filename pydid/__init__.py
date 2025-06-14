# pylint: disable=wildcard-import
"""DiD and doubly robust DiD estimators."""

# Bootstrapping functions
from pydid.drdid.boot.boot_ipw_rc import wboot_ipw_rc
from pydid.drdid.boot.boot_mult import mboot_did, mboot_twfep_did
from pydid.drdid.boot.boot_panel import (
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_ipw_panel,
    wboot_reg_panel,
    wboot_std_ipw_panel,
    wboot_twfe_panel,
)
from pydid.drdid.boot.boot_rc import wboot_drdid_rc1, wboot_drdid_rc2
from pydid.drdid.boot.boot_rc_ipt import wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2
from pydid.drdid.boot.boot_reg_rc import wboot_reg_rc
from pydid.drdid.boot.boot_std_ipw_rc import wboot_std_ipw_rc
from pydid.drdid.boot.boot_twfe_rc import wboot_twfe_rc

# DR-DiD estimators
from pydid.drdid.drdid_imp_local_rc import drdid_imp_local_rc
from pydid.drdid.drdid_imp_panel import drdid_imp_panel
from pydid.drdid.drdid_imp_rc import drdid_imp_rc
from pydid.drdid.drdid_panel import drdid_panel
from pydid.drdid.drdid_rc import drdid_rc
from pydid.drdid.drdid_trad_rc import drdid_trad_rc

# IPW DiD estimators
from pydid.drdid.ipw_did_panel import ipw_did_panel
from pydid.drdid.ipw_did_rc import ipw_did_rc

# Propensity score estimators
from pydid.drdid.propensity.aipw_estimators import aipw_did_panel, aipw_did_rc_imp1, aipw_did_rc_imp2
from pydid.drdid.propensity.ipw_estimators import ipw_rc
from pydid.drdid.propensity.pscore_ipt import calculate_pscore_ipt

# Outcome regression estimators
from pydid.drdid.reg_did_panel import reg_did_panel
from pydid.drdid.reg_did_rc import reg_did_rc
from pydid.drdid.std_ipw_did_panel import std_ipw_did_panel
from pydid.drdid.std_ipw_did_rc import std_ipw_did_rc

# Regression functions
from pydid.drdid.wols import wols_panel, wols_rc

# Panel data utilities
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
    "drdid_imp_panel",
    "drdid_imp_rc",
    "drdid_imp_local_rc",
    "drdid_panel",
    "drdid_rc",
    "drdid_trad_rc",
    # IPW DiD estimators
    "ipw_did_panel",
    "ipw_did_rc",
    "std_ipw_did_panel",
    "std_ipw_did_rc",
    # Outcome regression estimators
    "reg_did_panel",
    "reg_did_rc",
    # Core propensity score estimators
    "aipw_did_panel",
    "aipw_did_rc_imp1",
    "aipw_did_rc_imp2",
    "ipw_did_rc",
    "ipw_rc",
    "calculate_pscore_ipt",
    # Bootstrap functions
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
]
