.. _overview:

******************
What is ModernDiD?
******************

ModernDiD is a unified Python implementation of modern difference-in-differences
methodologies for causal inference. It is a library that consolidates
the fragmented landscape of DiD estimators from leading econometric research
and various R packages into a single, coherent framework with a consistent API,
robust inference, and native visualization.

At the core of the ModernDiD package is **causal inference under parallel trends**.
Difference-in-differences (DiD) is one of the most widely used methods for
estimating causal effects from observational data. The key insight is simple:
by comparing how outcomes change over time between treated and untreated groups,
we can isolate the effect of a treatment or policy, provided that both groups
would have followed parallel paths in the absence of treatment.

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

  import statsmodels.formula.api as smf
  model = smf.ols("outcome ~ treated + C(state) + C(year)", data=df)
  result = model.fit()

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


.. _whatis-modern:

Why modern DiD methods?
-----------------------

The methods in ModernDiD share several features that make them suitable
for credible causal inference. Each estimated effect corresponds to a
well-defined causal parameter: the average treatment effect on the treated
for a specific group at a specific time, using only valid comparison units.
Rather than assuming a single constant effect, these methods estimate
separate effects that can vary across groups and time, revealing important
patterns in how policies work.

The default estimators are doubly robust, combining propensity score
weighting with outcome regression. This provides consistency if *either*
the propensity score model *or* the outcome model is correctly specified,
offering protection against model misspecification. Standard errors account
for clustering at the appropriate level, and simultaneous confidence bands
adjust for multiple testing when examining many group-time effects or
event study coefficients.

Built-in sensitivity analysis tools assess how conclusions change under
plausible violations of parallel trends, using pre-treatment trends to
calibrate the degree of allowable violations.


.. _whatis-methods:

What methods does ModernDiD provide?
------------------------------------

ModernDiD consolidates several distinct methodological advances:

**Multi-period staggered DiD** (:mod:`~moderndid.did`) implements the
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
framework for estimating group-time average treatment effects with
staggered adoption, including aggregation to event studies, group
effects, and calendar time effects.

**Doubly robust two-period DiD** (:mod:`~moderndid.drdid`) provides the
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


.. _whatis-users:

Who uses DiD methods?
---------------------

Difference-in-differences is the workhorse of empirical economics and
increasingly important across the social sciences and related fields.
Researchers use these methods to evaluate policies
such as minimum wage increases, healthcare expansions, environmental
regulations, and tax reforms. Program evaluators measure the impacts of
job training programs, educational interventions, and public health
campaigns. Economic analysts understand the effects of pricing changes,
marketing campaigns, and operational improvements. Academic researchers
publish credible causal estimates in economics, political science,
sociology, epidemiology, and related fields.

The recent "credibility revolution" in empirical research has placed
increasing emphasis on transparent identification strategies, and DiD
methods, when properly implemented with modern techniques, remain one
of the most credible approaches available with observational data.

ModernDiD brings these methods to Python users in a unified framework,
enabling researchers and practitioners to conduct rigorous causal
inference with state-of-the-art tools.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   self
   installation
   project_philosophy
