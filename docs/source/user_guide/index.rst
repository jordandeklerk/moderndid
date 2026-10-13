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

.. admonition:: Start with a complete run
   :class: tip

   If you're new to ModernDiD, the :doc:`Start here <getting_started>` pages
   cover installation and walk you through a first analysis. When you have a
   specific question, such as which comparison group to use, the
   :doc:`FAQ <../faq>` gives a short answer and links to the guide that covers it.

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
through a complete analysis that explains the specification choices and checks
behind its conclusions.

The :doc:`Background <../background/index>` pages develop the assumptions and
theoretical results behind each estimator. Once you've chosen a design, the
:ref:`API Reference <api>` documents each estimator's arguments and result
fields.


.. raw:: html

   <p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
