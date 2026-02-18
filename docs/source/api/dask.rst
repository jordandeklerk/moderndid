.. _api-dask:

Distributed Computing (Dask)
=============================

The Dask module provides distributed implementations of the core estimators
for datasets that exceed single-machine memory. Most users should simply pass
a Dask DataFrame to :func:`~moderndid.att_gt` or :func:`~moderndid.ddd`,
which automatically dispatch to the distributed backend. The functions below
are the underlying implementations and are documented here for reference. For
practical usage guidance, see :doc:`/user_guide/distributed`.

Estimators
----------

.. currentmodule:: moderndid.dask

.. autosummary::
   :toctree: generated/dask/
   :nosignatures:

   dask_att_gt
   dask_ddd
