.. _api-data:

Datasets and simulation
=======================

The bundled datasets let you follow the worked examples or try an estimator
without preparing your own sample first. Each loader returns a Polars
DataFrame. The :doc:`data guide <../user_guide/data>` explains how to prepare
inputs for estimation.

When you need a known effect to check a calculation, the simulators generate
data for specific treatment designs. Tests and Monte Carlo studies use those
known effects to examine what an estimator recovers under the chosen design.

.. currentmodule:: moderndid

Built-in datasets
-----------------

.. autosummary::
   :toctree: generated/data/
   :nosignatures:

   load_mpdta
   load_nsw
   load_ehec
   load_engel
   load_favara_imbs
   load_fracking
   load_cai2016
   load_acemoglu

Simulation functions
--------------------

.. autosummary::
   :toctree: generated/data/
   :nosignatures:

   gen_cont_did_data
   gen_did_scalable
   gen_ddd_2periods
   gen_ddd_mult_periods
   gen_ddd_scalable
   gen_simple_ddd_data
