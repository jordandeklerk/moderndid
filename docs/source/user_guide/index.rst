.. _user-guide:
.. _overview:

==========
User Guide
==========

ModernDiD is a Python library for difference-in-differences (DiD) and related
causal inference methods. It is built for applied researchers using
observational data to study how policies and treatments affect outcomes.

Many of those studies follow units that begin treatment in different
periods. When effects differ across groups or change after adoption, a
:ref:`two-way fixed effects regression <background-did-twfe>` can combine
comparisons that give a misleading picture of the effects. ModernDiD's
staggered adoption estimators keep those effects separate so you can choose
how to average them for your research question. The
:doc:`estimator guide <estimator_overview>` covers the other designs the
package supports, including continuous doses, triple differences, and
treatment that can change over time.

We'll follow an analysis through the package's data preparation,
estimation, and reporting tools so you can see how they fit together.
The guides explain how to read the fitted effects and their uncertainty,
choose a summary, and present the results in figures and publication tables.
The guide also explains each method's identifying assumptions and inference
options so you can adapt the analysis to your own study.

If you're new to ModernDiD, begin with :doc:`getting_started` to try a
complete run. The :doc:`fundamentals` pages are also a reference you can
return to when you're working with your own data.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started
   fundamentals
   scaling

.. _whatis-design:

Find the guide you need
-----------------------

We've grouped these pages by where you are in an analysis to help you read
them in order or return to the decision you're working on.

.. list-table::
   :class: section-index-table
   :header-rows: 1
   :widths: 30 70

   * - Section
     - What you'll work through
   * - :doc:`getting_started`
     - Install the package, understand a DiD comparison, and run your first analysis.
   * - :doc:`fundamentals`
     - Choose an estimator, prepare data, read results, and produce figures and tables.
   * - :doc:`scaling`
     - Choose CPU parallelism or GPU acceleration for the work your estimator does.

.. _whatis-methods:

Follow an analysis or a derivation
----------------------------------

Each page in :doc:`Examples <../examples/index>` develops a research question
through a complete analysis, including the specification choices and checks
behind its conclusions.

The :doc:`Background <../background/index>` pages develop the assumptions and
theoretical results behind each estimator. The :ref:`API Reference <api>`
documents each estimator's arguments and result fields for checking the details
of a call once you've chosen the design.


.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
