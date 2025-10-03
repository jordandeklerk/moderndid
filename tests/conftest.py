"""Shared test configuration utilities for moderndid."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_ENV_FULL = "MODERNDID_RUN_FULL_TESTS"
_BASE_DIR = Path(__file__).resolve().parent
_SLOW_DIRS = {
    _BASE_DIR / "did",
    _BASE_DIR / "didcont",
    _BASE_DIR / "didhonest",
    _BASE_DIR / "drdid",
}

_DID_ALLOWLIST = {
    _BASE_DIR / "did" / "test_aggte.py",
    _BASE_DIR / "did" / "test_aggte_obj.py",
    _BASE_DIR / "did" / "test_att_gt.py",
    _BASE_DIR / "did" / "test_compute_aggte.py",
    _BASE_DIR / "did" / "test_compute_att_gt.py",
    _BASE_DIR / "did" / "test_preprocess_did.py",
    _BASE_DIR / "did" / "test_multiperiod_obj.py",
}

_DIDHONEST_ALLOWLIST = {
    _BASE_DIR / "didhonest" / "test_bounds.py",
    _BASE_DIR / "didhonest" / "test_conditional.py",
    _BASE_DIR / "didhonest" / "test_apr_nuisance.py",
    _BASE_DIR / "didhonest" / "test_fixed_length_ci.py",
    _BASE_DIR / "didhonest" / "test_utils.py",
}

_DRDID_ALLOWLIST = {
    _BASE_DIR / "drdid" / "test_drdid.py",
    _BASE_DIR / "drdid" / "test_ipwdid.py",
    _BASE_DIR / "drdid" / "test_ordid.py",
    _BASE_DIR / "drdid" / "test_estimators.py",
    _BASE_DIR / "drdid" / "test_drdid_imp_local_rc.py",
    _BASE_DIR / "drdid" / "test_drdid_imp_panel.py",
    _BASE_DIR / "drdid" / "test_drdid_imp_rc.py",
    _BASE_DIR / "drdid" / "test_drdid_panel.py",
    _BASE_DIR / "drdid" / "test_drdid_trad_rc.py",
    _BASE_DIR / "drdid" / "test_drdid_trad_rc_ipt.py",
    _BASE_DIR / "drdid" / "test_pcore_ipt.py",
}

_DIDCONT_ALLOWLIST = {
    _BASE_DIR / "didcont" / "test_bspline.py",
    _BASE_DIR / "didcont" / "test_confidence_bands.py",
    _BASE_DIR / "didcont" / "test_lepski.py",
    _BASE_DIR / "didcont" / "test_estimators.py",
    _BASE_DIR / "didcont" / "test_selection.py",
    _BASE_DIR / "didcont" / "test_prodspline.py",
    _BASE_DIR / "didcont" / "test_process_attgt.py",
    _BASE_DIR / "didcont" / "test_process_aggte.py",
    _BASE_DIR / "didcont" / "test_process_dose.py",
    _BASE_DIR / "didcont" / "test_process_panel.py",
    _BASE_DIR / "didcont" / "test_spline_base.py",
    _BASE_DIR / "didcont" / "test_cont_did.py",
    _BASE_DIR / "didcont" / "test_npiv.py",
}


def pytest_collection_modifyitems(items):
    """Skip very slow suites unless the full-test environment variable is set."""
    if os.environ.get(_ENV_FULL):
        return

    skip_marker = pytest.mark.skip(
        reason=(f"Skipped to keep the default CI test run fast. Set {_ENV_FULL}=1 to execute the full test battery.")
    )

    for item in items:
        path = Path(str(item.fspath)).resolve()

        if path.is_relative_to(_BASE_DIR / "did"):
            if path not in _DID_ALLOWLIST:
                item.add_marker(skip_marker)
            continue

        if path.is_relative_to(_BASE_DIR / "didhonest"):
            if path not in _DIDHONEST_ALLOWLIST:
                item.add_marker(skip_marker)
            continue

        if path.is_relative_to(_BASE_DIR / "drdid"):
            if path not in _DRDID_ALLOWLIST:
                item.add_marker(skip_marker)
            continue

        if path.is_relative_to(_BASE_DIR / "didcont"):
            if path not in _DIDCONT_ALLOWLIST:
                item.add_marker(skip_marker)
            continue

        if any(
            path.is_relative_to(slow_dir)
            for slow_dir in _SLOW_DIRS
            - {_BASE_DIR / "did", _BASE_DIR / "didhonest", _BASE_DIR / "drdid", _BASE_DIR / "didcont"}
        ):
            item.add_marker(skip_marker)
