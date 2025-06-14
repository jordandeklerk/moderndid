"""Doubly robust DiD estimators."""

# Bootstrapping functions
from .boot.boot_ipw_rc import wboot_ipw_rc
from .boot.boot_mult import mboot_did, mboot_twfep_did
from .boot.boot_panel import (
    wboot_dr_tr_panel,
    wboot_drdid_imp_panel,
    wboot_ipw_panel,
    wboot_reg_panel,
    wboot_std_ipw_panel,
    wboot_twfe_panel,
)
from .boot.boot_rc import wboot_drdid_rc1, wboot_drdid_rc2
from .boot.boot_rc_ipt import wboot_drdid_ipt_rc1, wboot_drdid_ipt_rc2
from .boot.boot_reg_rc import wboot_reg_rc
from .boot.boot_std_ipw_rc import wboot_std_ipw_rc
from .boot.boot_twfe_rc import wboot_twfe_rc

# DR-DiD estimators
from .estimators.drdid_imp_local_rc import drdid_imp_local_rc
from .estimators.drdid_imp_panel import drdid_imp_panel
from .estimators.drdid_imp_rc import drdid_imp_rc
from .estimators.drdid_panel import drdid_panel
from .estimators.drdid_rc import drdid_rc
from .estimators.drdid_trad_rc import drdid_trad_rc

# IPW DiD estimators
from .estimators.ipw_did_panel import ipw_did_panel
from .estimators.ipw_did_rc import ipw_did_rc
from .estimators.reg_did_panel import reg_did_panel
from .estimators.reg_did_rc import reg_did_rc
from .estimators.std_ipw_did_panel import std_ipw_did_panel
from .estimators.std_ipw_did_rc import std_ipw_did_rc
from .estimators.wols import wols_panel, wols_rc

# Outcome regression estimators
from .ordid import ordid

# Propensity score estimators
from .propensity.aipw_estimators import aipw_did_panel, aipw_did_rc_imp1, aipw_did_rc_imp2
from .propensity.ipw_estimators import ipw_rc
from .propensity.pscore_ipt import calculate_pscore_ipt

__all__ = [
    # DR-DiD estimators
    "drdid_imp_local_rc",
    "drdid_imp_panel",
    "drdid_imp_rc",
    "drdid_panel",
    "drdid_rc",
    "drdid_trad_rc",
    # IPW DiD estimators
    "ipw_did_panel",
    "ipw_did_rc",
    "std_ipw_did_panel",
    "std_ipw_did_rc",
    # Outcome regression estimators
    "ordid",
    "reg_did_panel",
    "reg_did_rc",
    # Propensity score estimators
    "aipw_did_panel",
    "aipw_did_rc_imp1",
    "aipw_did_rc_imp2",
    "calculate_pscore_ipt",
    "ipw_rc",
    # Bootstrapping functions
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
]
