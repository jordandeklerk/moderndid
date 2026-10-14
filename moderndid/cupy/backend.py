"""GPU backend dispatch for array operations."""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import re
import shutil
import subprocess
from contextvars import ContextVar

import numpy as np

try:
    import cupy as cp

    HAS_CUPY = True
    _CUPY_IMPORT_ERROR = None
except ImportError as err:
    HAS_CUPY = False
    cp = None
    # Since find_spec locates a package without running it, it tells a broken CuPy install from a missing one.
    _CUPY_IMPORT_ERROR = err if importlib.util.find_spec("cupy") is not None else None

__all__ = [
    "HAS_CUPY",
    "_array_module",
    "get_backend",
    "set_backend",
    "to_device",
    "to_numpy",
    "use_backend",
]

_active_backend: ContextVar[str] = ContextVar("moderndid_backend", default="numpy")


def set_backend(name):
    """Set the active array backend.

    Parameters
    ----------
    name : {"numpy", "cupy"}
        Backend to activate. Setting "cupy" requires CuPy to be installed.
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
    exception is raised. Since each ``copy_context()`` snapshot inherits the
    value set here, ``use_backend`` composes correctly with
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


def _array_module(*arrays):
    """Return the array module that matches the given arrays.

    Unlike :func:`get_backend`, which reads the active backend, this reads the
    arrays themselves. A helper that uses it always works in the module of the
    arrays it receives.

    Parameters
    ----------
    *arrays : array_like
        One or more arrays to inspect.

    Returns
    -------
    module
        ``cupy`` if any input is a CuPy ndarray, ``numpy`` otherwise.
    """
    if HAS_CUPY and hasattr(cp, "ndarray"):
        for arr in arrays:
            if isinstance(arr, cp.ndarray):
                return cp
    return np


@functools.lru_cache(1)
def _detect_cuda_version():
    """Try to detect the CUDA major version."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.splitlines():
            if "CUDA Version" in line:
                part = line.split("CUDA Version:")[-1].strip().rstrip("|").strip()
                return int(part.split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass

    if shutil.which("nvcc") is not None:
        try:
            out = subprocess.check_output(
                ["nvcc", "--version"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            m = re.search(r"release (\d+)\.", out)
            if m:
                return int(m.group(1))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    return None


def _cupy_install_message():
    """Build an actionable CuPy install error message."""
    if _CUPY_IMPORT_ERROR is not None:
        return (
            f"CuPy is installed but failed to import: {_CUPY_IMPORT_ERROR}\n"
            "Check that the CuPy wheel matches your CUDA version and that the environment holds only one CuPy package."
        )
    cuda_ver = _detect_cuda_version()
    if cuda_ver in (12, 13):
        return (
            f"CuPy is not installed. Detected CUDA {cuda_ver}.x on this machine.\n"
            "Install the matching wheel:\n"
            f'  uv add "cupy-cuda{cuda_ver}x[ctk]"\n'
            "Use pip install in place of uv add outside a uv project."
        )
    return (
        "CuPy is not installed. Install the wheel matching your CUDA version, which must be 12 or later "
        "(run 'nvidia-smi' or 'nvcc --version' to check):\n"
        '  uv add "cupy-cuda12x[ctk]"   # CUDA 12.x\n'
        '  uv add "cupy-cuda13x[ctk]"   # CUDA 13.x\n'
        "Use pip install in place of uv add outside a uv project."
    )


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
    """
    name = name.lower()
    if name not in ("numpy", "cupy"):
        raise ValueError(f"Unknown backend {name!r}. Choose 'numpy' or 'cupy'.")
    if name == "cupy" and not HAS_CUPY:
        raise ImportError(_cupy_install_message()) from _CUPY_IMPORT_ERROR
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
