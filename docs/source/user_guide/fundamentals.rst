.. _fundamentals:

============
Fundamentals
============

Your study may differ from the :ref:`first analysis <quickstart>` in its
treatment timing, in the completeness of its panel, or in the question it
asks. We'll work through what those differences mean for the estimator you
choose and the data it needs. Once you have an estimated result, the later
guides help you choose a summary of its effects and describe their uncertainty
in the figures and tables you report.

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
   * - :doc:`results`
     - Read result objects, choose an aggregation, and interpret uncertainty.
   * - :doc:`plotting`
     - Choose a figure for your result and adjust or save it.
   * - :doc:`publication_tables`
     - Present estimates and sample descriptions in tables for your paper.
   * - :doc:`panel_utilities`
     - Inspect duplicate observations, missing periods, and treatment histories.

The data, results, plotting, and tables guides all use the minimum wage county
data. Since the last three also fit the first analysis's specification, you can
follow one set of estimates from the printed report to a finished figure and
table. The guide on preparing panel data comes last and switches to a county
panel with missing years. There you can compare filling those gaps with
dropping the incomplete counties. When you want to examine the decisions behind
a complete specification, continue with the :ref:`Examples <examples>` for the
treatment design you're studying.

.. toctree::
   :hidden:
   :maxdepth: 1

   estimator_overview
   data
   results
   plotting
   publication_tables
   panel_utilities

.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
