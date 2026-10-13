.. _api-honestdid:

Honest DiD
==========

To examine how an event-study conclusion depends on parallel trends,
:func:`~moderndid.honest_did` builds confidence sets that allow deviations
within limits you choose. The method follows
`Rambachan and Roth (2023) <https://arxiv.org/abs/2203.04511>`_. Its lower-level
functions construct sets for restrictions on the size or smoothness of
those deviations, including combinations of the two.

The :ref:`sensitivity analysis example <example_honest_did>` works through
the Medicaid expansion data. The :ref:`background page <background-didhonest>`
explains how each restriction changes the conclusions the data support.

.. admonition:: The restriction determines the robustness claim
   :class: important

   Coverage depends on the chosen restriction holding. A confidence set for
   bounded deviations does not establish robustness to unrestricted failures
   of parallel trends.

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
