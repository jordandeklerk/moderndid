"""Test related helper functions based on arviz-stats."""

import os
import sys
import warnings
from typing import Any

import pytest


def importorskip(modname: str, reason: str | None = None) -> Any:
    """Import and return the requested module ``modname``.

    Doesn't allow skips when ``DRSYNTHD_REQUIRE_ALL_DEPS`` env var is defined.
    Borrowed and modified from ``pytest.importorskip``.

    Parameters
    ----------
    modname : str
        the name of the module to import
    reason : str, optional
        this reason is shown as skip message when the module cannot be imported.
    """
    __tracebackhide__ = True  # pylint: disable=unused-variable
    compile(modname, "", "eval")  # to catch syntaxerrors

    with warnings.catch_warnings():
        # Make sure to ignore ImportWarnings that might happen because
        # of existing directories with the same name we're trying to
        # import but without a __init__.py file.
        warnings.simplefilter("ignore")
        try:
            __import__(modname)
        except ImportError as exc:
            if "DRSYNTHD_REQUIRE_ALL_DEPS" in os.environ:
                raise exc
            if reason is None:
                reason = f"could not import {modname!r}: {exc}"
            pytest.skip(reason, allow_module_level=True)

    mod = sys.modules[modname]
    return mod
