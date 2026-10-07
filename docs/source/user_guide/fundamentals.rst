============
Fundamentals
============

Once you've run the :doc:`quickstart`, the next task is to adapt that analysis
to your study. We start by matching an estimator to your treatment history
and checking that the data support its comparisons. After fitting the model,
you'll need to decide which effects to average and how to describe their
uncertainty before presenting them in figures and publication tables.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Guide
     - What you'll work through
   * - :doc:`estimator_overview`
     - Match the treatment path and causal question to a supported estimator.
   * - :doc:`data`
     - Organize identifiers, treatment timing, covariates, and sampling weights.
   * - :doc:`panel_utilities`
     - Inspect duplicate observations, missing periods, and treatment histories.
   * - :doc:`results`
     - Read result objects, choose an aggregation, and interpret uncertainty.
   * - :doc:`plotting`
     - Choose a figure for your result and adjust or save it.
   * - :doc:`publication_tables`
     - Present estimates and sample descriptions in tables for your paper.

These guides use county data to make each choice concrete, from deciding
what a row represents to explaining an effect in a publication table. As you
work through them, you can carry the same questions back to your own data.
The :doc:`Examples <../examples/index>` bring these building blocks together
in full applications once you're ready to examine a specification and the
checks behind its conclusions.

.. toctree::
   :hidden:
   :maxdepth: 1

   estimator_overview
   data
   panel_utilities
   results
   plotting
   publication_tables

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
