.. _api-honestdid:

Honest DiD
==========

Following `Rambachan and Roth (2023) <https://arxiv.org/abs/2203.04511>`_,
:func:`~moderndid.honest_did` builds confidence sets for event-study estimates
that stay valid when parallel trends fail within limits you choose. The
functions below it compute those confidence sets for each kind of restriction
on the violations. See the :ref:`sensitivity analysis example <example_honest_did>`
for a full analysis of the Medicaid expansion data and the
:ref:`background page <background-didhonest>` for the restrictions.

.. currentmodule:: moderndid

Main functions
--------------

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   honest_did
   construct_original_cs
   create_sensitivity_results_rm
   create_sensitivity_results_sm

Confidence intervals
--------------------

ARP confidence intervals
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_arp_ci
   compute_arp_nuisance_ci
   compute_least_favorable_cv
   compute_vlo_vup_dual
   lp_conditional_test
   test_in_identified_set
   test_in_identified_set_flci_hybrid
   test_in_identified_set_lf_hybrid

Fixed-length confidence intervals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_flci
   folded_normal_quantile
   maximize_bias
   minimize_variance

Restriction types
-----------------

Relative magnitude restrictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_rm
   compute_identified_set_rm
   compute_conditional_cs_rmb
   compute_identified_set_rmb
   compute_conditional_cs_rmm
   compute_identified_set_rmm

Smoothness restrictions
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_sd
   compute_identified_set_sd
   compute_conditional_cs_sdb
   compute_identified_set_sdb
   compute_conditional_cs_sdm
   compute_identified_set_sdm

Combined restrictions
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   compute_conditional_cs_sdrm
   compute_identified_set_sdrm
   compute_conditional_cs_sdrmb
   compute_identified_set_sdrmb
   compute_conditional_cs_sdrmm
   compute_identified_set_sdrmm

Result objects
--------------

.. autosummary::
   :toctree: generated/honestdid/
   :nosignatures:

   HonestDiDResult
   OriginalCSResult
   SensitivityResult
   FLCIResult
   APRCIResult
   ARPNuisanceCIResult
