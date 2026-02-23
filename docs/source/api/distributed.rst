.. _api-dask:

.. _api-spark:

Distributed
===========

The distributed modules provide Dask and Spark implementations of the core
estimators for datasets that exceed single-machine memory. Most users should
simply pass a Dask or PySpark DataFrame to :func:`~moderndid.att_gt` or
:func:`~moderndid.ddd`, which automatically dispatch to the appropriate
backend. The functions below are the underlying implementations and are
documented here for reference. For practical usage guidance, see
:doc:`/user_guide/distributed`.

Dask
----

.. currentmodule:: moderndid.dask

.. autosummary::
   :toctree: generated/dask/
   :nosignatures:

   dask_att_gt
   dask_ddd

Spark
-----

.. currentmodule:: moderndid.spark

.. autosummary::
   :toctree: generated/spark/
   :nosignatures:

   spark_att_gt
   spark_ddd
