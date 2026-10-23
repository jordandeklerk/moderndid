.. _user-guide:
.. _overview:

==========
User Guide
==========

ModernDiD is a Python library for difference-in-differences (DiD) and related
causal inference methods. It is built for applied researchers using
observational data to study how policies and treatments affect outcomes.

We'll follow one study through most of the guides, from preparing the data to
estimating its effects and presenting them in figures and publication tables.
The study asks how state minimum wage increases affected teen employment.
Along the way the guides explain how to read the fitted effects and their
uncertainty, what each method assumes, and which inference options you can
change for your own study.

.. _whatis-design:

Getting started
---------------

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Guide
     - What you'll work through
   * - :doc:`Installation <installation>`
     - Set up Python and install the dependencies your analysis needs.
   * - :doc:`DiD introduction <causal_inference>`
     - Understand the untreated comparison, parallel trends, and staggered adoption.
   * - :doc:`First analysis <quickstart>`
     - Estimate employment effects, summarize them, and draw an event study.

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation <installation>
   DiD introduction <causal_inference>
   First analysis <quickstart>

.. _fundamentals:

Data and results
----------------

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Guide
     - What you'll work through
   * - :doc:`Choosing an estimator <estimator_overview>`
     - Match the treatment path and causal question to a supported estimator.
   * - :doc:`Data for estimation <data>`
     - Organize identifiers, treatment timing, covariates, and sampling weights.
   * - :doc:`Working with estimates <results>`
     - Read result objects, choose an aggregation, and interpret uncertainty.
   * - :doc:`Plotting <plotting>`
     - Choose a figure for your result and adjust or save it.
   * - :doc:`Publication tables <publication_tables>`
     - Present estimates and sample descriptions in tables for your paper.
   * - :doc:`Preparing panel data <panel_utilities>`
     - Inspect duplicate observations, missing periods, and treatment histories.

.. toctree::
   :hidden:
   :maxdepth: 1

   Choosing an estimator <estimator_overview>
   Data for estimation <data>
   Working with estimates <results>
   Plotting <plotting>
   Publication tables <publication_tables>
   Preparing panel data <panel_utilities>

Scaling
-------

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Guide
     - What you'll work through
   * - :doc:`CPU parallelism <scaling>`
     - Spread comparisons and bootstrap draws over several CPU workers.
   * - :doc:`GPU acceleration <gpu>`
     - Set up CuPy, select supported GPU calculations, and measure their running time.
   * - :doc:`Setting up Spark <spark_setup>`
     - Install ModernDiD for Spark on Databricks, a cluster, or one machine and fix setup errors.
   * - :doc:`Distributed estimation on Spark <spark>`
     - Fit group-time effects on a Spark DataFrame and check them against an in-memory fit.

.. toctree::
   :hidden:
   :maxdepth: 1

   CPU parallelism <scaling>
   GPU acceleration <gpu>
   Setting up Spark <spark_setup>
   Distributed estimation on Spark <spark>

.. _whatis-methods:

Follow an analysis or a derivation
----------------------------------

Each page in :doc:`Examples <../examples/index>` develops a research question
through a complete analysis that explains the specification choices and checks
behind its conclusions.

The :doc:`Background <../background/index>` pages develop the assumptions and
theoretical results behind each estimator. Once you've chosen a design, the
:ref:`API Reference <api>` documents each estimator's arguments and result
fields.

When you have a specific question, such as which comparison group to use, the
:doc:`FAQ <../faq>` gives a short answer and links to the guide that covers it.


.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
