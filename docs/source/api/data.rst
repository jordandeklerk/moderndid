.. _api-data:

Datasets and simulation
=======================

Each loader below returns one of the datasets that come with ModernDiD as a
polars DataFrame. The simulators generate data with known treatment effects for
each design. Monte Carlo studies and tests use them to check how well an
estimator recovers those effects.

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
