"""GPU backend dispatch for array operations."""

from __future__ import annotations

import contextlib
import functools
import shutil
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
    "_array_module",
    "get_backend",
    "set_backend",
    "to_device",
    "to_numpy",
    "use_backend",
]

_active_backend: ContextVar[str] = ContextVar("moderndid_backend", default="numpy")
_rmm_initialized = False


def set_backend(name):
    """Set the active array backend.

    Parameters
    ----------
    name : {"numpy", "cupy"}
        Backend to activate. Setting "cupy" requires CuPy to be installed.
    """
    name = _validate_backend_name(name)
    if name == "cupy":
        _init_rmm_pool()
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
    if name == "cupy":
        _init_rmm_pool()
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
    """Return ``cupy`` if any array is a CuPy ndarray, else ``numpy``.

    This is used by Dask partition functions to detect whether their
    input arrays are CuPy or NumPy and use the matching ``xp`` module,
    without relying on the ``ContextVar``-based ``get_backend()`` which
    does not propagate to Dask worker processes.

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
    import re
    import subprocess

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
    except (OSError, ValueError):
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
        except (OSError, ValueError):
            pass

    return None


def _cupy_install_message():
    """Build an actionable CuPy install error message."""
    cuda_ver = _detect_cuda_version()
    if cuda_ver is not None:
        wheel = f"cupy-cuda{cuda_ver}x"
        return (
            f"CuPy is not installed. Detected CUDA {cuda_ver}.x on this machine.\n"
            f"Install the matching wheel:\n"
            f"  uv pip install {wheel}"
        )
    return (
        "CuPy is not installed. Install the wheel matching your CUDA version "
        "(run 'nvidia-smi' or 'nvcc --version' to check):\n"
        "  uv pip install cupy-cuda11x   # CUDA 11.x\n"
        "  uv pip install cupy-cuda12x   # CUDA 12.x"
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
        raise ImportError(_cupy_install_message())
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


def _init_rmm_pool():
    """Activate the RMM memory-pool allocator for CuPy.

    When ``rmm`` is installed, this replaces CuPy's default per-allocation
    ``cudaMalloc`` calls with a pool allocator that grows on demand, which
    eliminates the ~1 ms overhead per allocation that otherwise dominates
    bootstrap loops.  If ``rmm`` is not installed, this is a silent no-op.

    If CuPy is already using the RMM allocator (for example, because the
    user configured RMM manually before importing ModernDiD), this function
    is a no-op and the existing configuration is preserved.
    """
    global _rmm_initialized
    if _rmm_initialized:
        return
    try:
        from rmm.allocators.cupy import rmm_cupy_allocator

        if cp.cuda.get_allocator() == rmm_cupy_allocator:
            _rmm_initialized = True
            return

        import rmm

        rmm.reinitialize(pool_allocator=True, initial_pool_size=0)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        _rmm_initialized = True
    except Exception:  # noqa: BLE001
        pass
