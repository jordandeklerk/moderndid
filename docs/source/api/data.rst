.. _api-data:

Datasets and Simulation
========================

ModernDiD includes built-in datasets for examples and benchmarking, as well as
data generation functions for simulation studies and scalability testing.

Built-in Datasets
-----------------

.. currentmodule:: moderndid

.. autosummary::
   :toctree: generated/data/
   :nosignatures:

   load_mpdta
   load_nsw
   load_ehec
   load_engel
   load_favara_imbs
   load_cai2016

Simulation Functions
--------------------

Functions for generating synthetic DiD and DDD panel data with known treatment
effects, useful for Monte Carlo experiments and testing.

.. autosummary::
   :toctree: generated/data/
   :nosignatures:

   simulate_cont_did_data
   gen_did_scalable
   gen_dgp_2periods
   gen_dgp_mult_periods
   gen_dgp_scalable
   generate_simple_ddd_data
