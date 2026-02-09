"""GPU backend dispatch for array operations."""

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
]

_active_backend = "numpy"


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
    """
    global _active_backend
    name = name.lower()
    if name not in ("numpy", "cupy"):
        raise ValueError(f"Unknown backend {name!r}. Choose 'numpy' or 'cupy'.")
    if name == "cupy" and not HAS_CUPY:
        raise ImportError("CuPy is not installed. Install with: uv pip install 'moderndid[gpu]'")
    if name == "cupy":
        _gpu_ok = False
        if hasattr(cp, "is_available"):
            _gpu_ok = cp.is_available()
        else:
            try:
                cp.array([1.0])
                _gpu_ok = True
            except (RuntimeError, AttributeError):
                _gpu_ok = False
        if not _gpu_ok:
            raise RuntimeError(
                "CuPy is installed but no CUDA GPU is available. Check your CUDA installation or use backend='numpy'."
            )
    _active_backend = name


def get_backend():
    """Return the active array module (``numpy`` or ``cupy``).

    Returns
    -------
    module
        ``numpy`` when the backend is "numpy", ``cupy`` when "cupy".
    """
    if _active_backend == "cupy":
        return cp
    return np


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
