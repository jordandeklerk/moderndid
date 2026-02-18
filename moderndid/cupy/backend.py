"""GPU backend dispatch for array operations."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

__all__ = [
    "HAS_CUPY",
    "get_backend",
    "set_backend",
    "to_device",
    "to_numpy",
    "use_backend",
]

_active_backend: ContextVar[str] = ContextVar("moderndid_backend", default="numpy")


def _validate_backend_name(name):
    """Validate and normalise a backend name.

    Parameters
    ----------
    name : str
        Backend name (case-insensitive).

    Returns
    -------
    str
        Normalised backend name (``"numpy"`` or ``"cupy"``).

    Raises
    ------
    ValueError
        If *name* is not a recognised backend.
    ImportError
        If ``"cupy"`` is requested but CuPy is not installed.
    RuntimeError
        If ``"cupy"`` is requested but no CUDA GPU is available.
    """
    name = name.lower()
    if name not in ("numpy", "cupy"):
        raise ValueError(f"Unknown backend {name!r}. Choose 'numpy' or 'cupy'.")
    if name == "cupy" and not HAS_CUPY:
        raise ImportError("CuPy is not installed. Install with: uv pip install 'moderndid[gpu]'")
    if name == "cupy":
        _cupy_ok = False
        if hasattr(cp, "is_available"):
            _cupy_ok = cp.is_available()
        else:
            try:
                cp.array([1.0])
                _cupy_ok = True
            except (RuntimeError, AttributeError):
                _cupy_ok = False
        if not _cupy_ok:
            raise RuntimeError(
                "CuPy is installed but no CUDA GPU is available. Check your CUDA installation or use backend='numpy'."
            )
    return name


def set_backend(name):
    """Set the active array backend.

    Parameters
    ----------
    name : {"numpy", "cupy"}
        Backend to activate. Setting "cupy" requires CuPy to be installed.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend.
    ImportError
        If "cupy" is requested but CuPy is not installed.
    RuntimeError
        If "cupy" is requested but no CUDA GPU is available.
    """
    name = _validate_backend_name(name)
    _active_backend.set(name)


def get_backend():
    """Return the active array module (``numpy`` or ``cupy``).

    Returns
    -------
    module
        ``numpy`` when the backend is "numpy", ``cupy`` when "cupy".
    """
    if _active_backend.get() == "cupy":
        return cp
    return np


@contextlib.contextmanager
def use_backend(name):
    """Context manager that temporarily activates a backend.

    The previous backend is restored when the context exits, even if an
    exception is raised.  Each ``copy_context()`` snapshot inherits the
    value set here, so ``use_backend`` composes correctly with
    :func:`~moderndid.core.parallel.parallel_map`.

    Parameters
    ----------
    name : {"numpy", "cupy"}
        Backend to activate for the duration of the block.
    """
    name = _validate_backend_name(name)
    token = _active_backend.set(name)
    try:
        yield
    finally:
        _active_backend.reset(token)


def to_device(arr):
    """Move an array to the active device.

    Parameters
    ----------
    arr : array_like
        Input array (NumPy or CuPy).

    Returns
    -------
    ndarray
        Array on the active device.
    """
    xp = get_backend()
    if xp is cp:
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr)
        return arr
    # numpy backend — ensure we have a NumPy array
    if HAS_CUPY and hasattr(cp, "ndarray") and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_numpy(arr):
    """Ensure the array is a CPU NumPy array.

    Parameters
    ----------
    arr : array_like
        Input array (NumPy or CuPy).

    Returns
    -------
    numpy.ndarray
        CPU NumPy array.
    """
    if HAS_CUPY and hasattr(cp, "ndarray") and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
