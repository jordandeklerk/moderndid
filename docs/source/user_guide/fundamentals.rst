============
Fundamentals
============

Your study may have different treatment timing, an incomplete panel, or a
question that the :doc:`first analysis <quickstart>` doesn't address. We'll
work through what those differences mean for the estimator you choose and
the data it needs.
You can then use its results to construct a summary of the effects and
describe their uncertainty in the figures and tables you report.

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

These guides keep using the county data so you can follow familiar variables
from preparation into an estimated effect and its presentation. When you want
to examine the decisions behind a complete specification, continue with the
:doc:`Examples <../examples/index>` for the treatment design you're studying.

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
