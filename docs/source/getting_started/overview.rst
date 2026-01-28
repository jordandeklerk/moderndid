.. _overview:

******************
What is ModernDiD?
******************

ModernDiD is a unified Python implementation of modern difference-in-differences
methodologies for causal inference. It consolidates the fragmented landscape of
DiD estimators from leading econometric research into a single, coherent
framework with a consistent API, robust inference, and native visualization.

At the core of ModernDiD is **causal inference under parallel trends**.
Difference-in-differences (DiD) is one of the most widely used methods for
estimating causal effects from observational data. The method compares how
outcomes change over time between treated and untreated groups to isolate
the effect of a treatment or policy, provided that both groups would have
followed parallel paths in the absence of treatment.

There are several important reasons why traditional approaches to DiD
(such as two-way fixed effects regression) can produce misleading results
in modern settings:

- When units receive treatment at different times (staggered adoption),
  traditional methods implicitly use already-treated units as controls.
  This leads to *negative weighting* of some treatment effects, potentially
  producing estimates with the wrong sign even when all true effects are positive.

- Treatment effects often vary across groups and over time. Traditional
  methods assume a single constant effect, which masks important
  heterogeneity and can produce estimates that do not correspond to any
  meaningful causal parameter.

- When parallel trends may not hold exactly, researchers need tools to
  assess how sensitive their conclusions are to violations of this
  assumption. Traditional methods offer no systematic way to conduct
  such sensitivity analysis.

- Modern research designs often involve continuous treatments, triple
  differences, or repeated cross-sections rather than balanced panels.
  These settings require specialized methods that maintain the credibility
  of causal inference.

ModernDiD addresses these challenges by implementing state-of-the-art methods
from the econometrics literature. As a simple example, consider estimating the
effect of a policy that different states adopted at different times. With
traditional two-way fixed effects, we might write::

  import pyfixest as pf
  model = pf.feols("outcome ~ treated | state + year", data=df)

This produces a single coefficient, but that coefficient is a weighted average
of many underlying comparisons, some of which use already-treated states as
controls with negative weights. If treatment effects grow over time, these
negative weights can cause the estimate to be biased toward zero or even
negative when all true effects are positive.

ModernDiD provides methods that avoid these pitfalls::

  import moderndid as did

  result = did.att_gt(
      data=df,
      yname="outcome",
      tname="year",
      idname="state",
      gname="first_treated",
  )

This estimates separate treatment effects for each combination of treatment
cohort (group) and time period. These *group-time average treatment effects*,
or ATT(g,t), represent clean causal comparisons that use only never-treated
or not-yet-treated units as controls. They can then be aggregated into
interpretable summaries such as event studies showing how effects evolve
relative to treatment timing.


.. _whatis-unified:

Why a unified package?
----------------------

Over the past decade, econometricians have developed substantial improvements
to the classical two-way fixed effects approach. Yet Python users have been
largely excluded from this methodological progress. The modern DiD literature
has produced excellent R and Stata packages, but these implementations remain
fragmented across dozens of separate libraries, each with its own API, data
requirements, and output formats.

Each methodological paper typically produces its own package.
`did <https://bcallaway11.github.io/did/>`_ implements
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_,
`DRDID <https://pedrohcgs.github.io/DRDID/>`_ implements
`Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_,
`HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_ implements
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_,
and `contdid <https://github.com/bcallaway11/contdid>`_ implements
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_.
Researchers must learn multiple APIs, convert data between formats,
and reconcile different output structures when combining methods.

ModernDiD consolidates these methods into a single package with a consistent
interface. The same parameter names (``yname``, ``tname``, ``idname``,
``gname``, and more) work across all estimators. Result objects share common structures.
Plotting functions accept outputs from any module. This consistency reduces
the cognitive load of switching between methods and makes it easier to compare
results across different estimation strategies.


.. _whatis-methods:

What methods does ModernDiD provide?
------------------------------------

ModernDiD consolidates several distinct methodological advances.

**Multi-period staggered DiD** (:mod:`~moderndid.did`) implements the
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
framework for estimating group-time average treatment effects with
staggered adoption, including aggregation to event studies, group
effects, and calendar time effects.

**Doubly robust two-period DiD** (``drdid``) provides the
`Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_
estimators for classic two-period, two-group settings, with options
for inverse probability weighting, outcome regression, or doubly
robust combinations.

**Continuous treatment DiD** (:mod:`~moderndid.didcont`) extends the
framework to settings with continuous treatment intensity using the
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_
methods, producing dose-response functions that show how effects vary
with treatment dose.

**Triple difference-in-differences** (:mod:`~moderndid.didtriple`)
implements the `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
methods that leverage a third dimension of variation (such as eligibility
status) to relax parallel trends assumptions.

**Sensitivity analysis** (:mod:`~moderndid.didhonest`) provides the
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_
tools for robust inference under violations of parallel trends, producing
confidence intervals that remain valid under specified degrees of
assumption violation.


.. _whatis-performance:

Performance
-----------

Causal inference often involves large datasets and computationally intensive
procedures like bootstrapping. ModernDiD is built for performance from the
ground up.

Data wrangling uses `Polars <https://pola.rs/>`_ internally, providing
substantial speed improvements over pandas for the grouping, filtering, and
reshaping operations common in panel data analysis. Users can pass pandas
DataFrames directly and conversion happens automatically.

Numerical computations use NumPy with vectorized operations wherever possible.
Performance-critical inner loops, particularly in bootstrap procedures, use
`Numba <https://numba.pydata.org/>`_ JIT compilation to achieve near-C speeds
while maintaining readable Python code.


.. _whatis-design:

Design
------

ModernDiD follows several principles that guide development decisions.

**Correctness.** Every estimator is validated against reference
implementations from the original R packages. Test suites include numerical
comparisons ensuring that ModernDiD produces the same point estimates,
standard errors, and confidence intervals as established implementations.

**Sensible defaults.** The default options reflect best practices from the
methodological literature. Doubly robust estimation is the default because
it provides protection against misspecification. Never-treated units serve
as the default control group because this avoids contamination from
already-treated units. Users can override these defaults, but the out-of-box
experience should produce credible results.

**Transparency.** Result objects include not just point estimates
but also influence functions, variance-covariance matrices, and estimation
metadata. Users can inspect exactly how estimates were computed and use
intermediate outputs for custom analyses. Warning messages explain when
data issues might affect results.

**Interoperability.** ModernDiD works with the Python data science ecosystem.
Input data can be pandas or Polars DataFrames. Outputs are NumPy arrays and
named tuples that integrate with standard workflows. Plots use plotnine,
allowing full customization through the grammar of graphics.

Modern causal inference methods should be accessible to all empirical
researchers regardless of their preferred programming language.
ModernDiD brings these methods to the Python ecosystem.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   self
   installation
   causal_inference
   quickstart
