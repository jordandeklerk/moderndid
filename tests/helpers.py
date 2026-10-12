"""Test-related helper functions based on arviz-stats."""

import os
import sys
import warnings

import pytest


def importorskip(modname, reason=None):
    """Import and return the requested module ``modname``.

    Raise the import error instead of skipping when the ``MODERNDID_REQUIRE_ALL_DEPS``
    environment variable is set. Borrowed and modified from ``pytest.importorskip``.

    Parameters
    ----------
    modname : str
        Name of the module to import.
    reason : str, optional
        Skip message shown when the module cannot be imported.

    Returns
    -------
    module
        The imported module.
    """
    __tracebackhide__ = True
    compile(modname, "", "eval")  # to catch syntax errors

    with warnings.catch_warnings():
        # Make sure to ignore ImportWarnings that might happen because
        # of existing directories with the same name we're trying to
        # import but without a __init__.py file.
        warnings.simplefilter("ignore", ImportWarning)
        try:
            __import__(modname)
        except ImportError as exc:
            if "MODERNDID_REQUIRE_ALL_DEPS" in os.environ:
                raise exc
            if reason is None:
                reason = f"could not import {modname!r}: {exc}"
            pytest.skip(reason, allow_module_level=True)

    mod = sys.modules[modname]
    return mod
