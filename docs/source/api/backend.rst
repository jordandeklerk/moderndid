.. _api-backend:

GPU backend
===========

Supported estimators compute with NumPy on the CPU by default and can switch to
CuPy on an NVIDIA GPU. Use :func:`~moderndid.set_backend` to switch for the rest
of a session or :func:`~moderndid.use_backend` to switch inside a ``with`` block
only. To see which one is active, :func:`~moderndid.get_backend` returns the
array module in use. The :ref:`GPU guide <gpu>` covers installing CuPy and which
computations run on the GPU.

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/backend/
   :nosignatures:

   set_backend
   use_backend
   get_backend

Data transfer
-------------

The estimators move their arrays to the active device and back on their own.
These helpers do the same for arrays you handle yourself.

.. currentmodule:: moderndid.cupy

.. autosummary::
   :toctree: generated/backend/
   :nosignatures:

   to_device
   to_numpy
