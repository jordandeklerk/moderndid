.. _api-backend:

GPU backend
===========

Supported estimators use NumPy on the CPU by default. To run their supported
array calculations on an NVIDIA GPU, select CuPy with
:func:`~moderndid.set_backend` or :func:`~moderndid.use_backend`. Choose the
context manager when the setting should apply to one ``with`` block; use
``set_backend`` when subsequent calls should keep using the selected backend.
:func:`~moderndid.get_backend` returns the active array module.

.. admonition:: GPU support depends on the calculation
   :class: note

   Selecting CuPy does not move every part of an analysis to the GPU. The
   :ref:`GPU guide <gpu>` explains which estimators and computations support
   it and how to install the required dependencies.

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/backend/
   :nosignatures:

   set_backend
   use_backend
   get_backend

Data transfer
-------------

Supported estimators handle transfers between CPU and GPU storage internally.
When you work with arrays directly, these helpers move them to the active
device or return them to NumPy for further CPU calculations.

.. currentmodule:: moderndid.cupy

.. autosummary::
   :toctree: generated/backend/
   :nosignatures:

   to_device
   to_numpy
