====================
ModernDiD Quickstart
====================

Prerequisites
==============

You'll need basic familiarity with Python. Experience with pandas or Polars
DataFrames is helpful but not required. ModernDiD accepts either pandas or
Polars DataFrames as input, and the basic API is straightforward enough for
Python beginners to use.

This guide assumes you are familiar with the difference-in-differences
methodology, particularly the `Callaway and Sant'Anna (2021)
<https://arxiv.org/abs/1803.09015>`_ framework for staggered adoption
designs. We focus on practical usage rather than theory. For a high-level
theoretical overview, see the :ref:`background` section.

**Learner profile**

This is a quick overview of difference-in-differences estimation in ModernDiD.
It demonstrates how to estimate treatment effects when units receive treatment
at different times (staggered adoption), how to aggregate results into event
studies, and how to visualize the findings.

**Learning Objectives**

After reading, you should be able to:

- Load data and identify the key variables needed for DiD estimation
- Estimate group-time average treatment effects using ``att_gt``
- Aggregate effects into event studies and other summaries using ``aggte``
- Visualize results with built-in plotting functions
- Configure bootstrap inference, anticipation periods, and other options


The basics
==========

ModernDiD estimates causal effects by comparing how outcomes change over time
between treated and untreated units. The fundamental building block is the
**group-time average treatment effect**, or ATT(g,t), which measures the
average effect of treatment for units first treated in period *g*, evaluated
at time *t*.

A **group** is defined by when units first received treatment. For example,
if some states raised their minimum wage in 2010 and others in 2012, you have
two treatment groups, the 2010 group and the 2012 group. Units that never
receive treatment form the comparison group.

The key variables you need are:

``yname``
    The outcome variable (e.g., employment, earnings, test scores).

``tname``
    The time period variable (e.g., year, quarter, month).

``idname``
    The unit identifier (e.g., state, county, individual ID).

``gname``
    The group variable indicating when a unit was first treated,
    with 0 for never-treated units.

This quickstart focuses on the :mod:`~moderndid.did` subpackage, which
implements the `Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
estimator for staggered adoption designs. All estimators in ModernDiD
share a consistent API, with the same core arguments (``yname``,
``tname``, ``idname``) appearing across methods. Options like ``xformla``
for covariates, ``est_method`` for estimation approach, and ``bstrap`` for
bootstrap inference work similarly throughout. Once you learn one estimator,
the others follow the same patterns with method-specific extensions.


Loading data
------------

ModernDiD includes several example datasets for experimentation. Here
we use county-level teen employment data where states adopted minimum
wage increases at different times:

.. code-block:: python

    import moderndid as did

    data = did.load_mpdta()
    print(data.head())

.. code-block:: text

    shape: (5, 6)
    ┌──────┬────────────┬──────────┬──────────┬─────────────┬───────┐
    │ year ┆ countyreal ┆ lpop     ┆ lemp     ┆ first.treat ┆ treat │
    │ ---  ┆ ---        ┆ ---      ┆ ---      ┆ ---         ┆ ---   │
    │ i64  ┆ i64        ┆ f64      ┆ f64      ┆ i64         ┆ i64   │
    ╞══════╪════════════╪══════════╪══════════╪═════════════╪═══════╡
    │ 2003 ┆ 8001       ┆ 5.896761 ┆ 8.461469 ┆ 2007        ┆ 1     │
    │ 2004 ┆ 8001       ┆ 5.896761 ┆ 8.33687  ┆ 2007        ┆ 1     │
    │ 2005 ┆ 8001       ┆ 5.896761 ┆ 8.340217 ┆ 2007        ┆ 1     │
    │ 2006 ┆ 8001       ┆ 5.896761 ┆ 8.378161 ┆ 2007        ┆ 1     │
    │ 2007 ┆ 8001       ┆ 5.896761 ┆ 8.487352 ┆ 2007        ┆ 1     │
    └──────┴────────────┴──────────┴──────────┴─────────────┴───────┘

The dataset contains 500 counties observed over 2003-2007. The key
variables are ``lemp`` (log employment), ``year``, ``countyreal``
(county identifier), and ``first.treat`` (the year a county was first
treated, or 0 for never-treated).


Estimating group-time effects
-----------------------------

The :func:`~moderndid.did.att_gt` function estimates treatment effects
for each combination of treatment group and time period:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
    )
    print(result)

.. code-block:: text

    Reference: Callaway and Sant'Anna (2021)

    Group-Time Average Treatment Effects:
      Group   Time   ATT(g,t)   Std. Error  [95% Simult. Conf. Band]
       2004   2004    -0.0105       0.0240  [ -0.0768,   0.0558]
       2004   2005    -0.0704       0.0292  [ -0.1511,   0.0103]
       2004   2006    -0.1373       0.0377  [ -0.2415,  -0.0330] *
       2004   2007    -0.1008       0.0342  [ -0.1954,  -0.0062] *
       2006   2004     0.0065       0.0233  [ -0.0580,   0.0711]
       2006   2005    -0.0028       0.0204  [ -0.0592,   0.0537]
       2006   2006    -0.0046       0.0178  [ -0.0539,   0.0447]
       2006   2007    -0.0412       0.0204  [ -0.0977,   0.0152]
       2007   2004     0.0305       0.0153  [ -0.0119,   0.0729]
       2007   2005    -0.0027       0.0162  [ -0.0475,   0.0421]
       2007   2006    -0.0311       0.0166  [ -0.0770,   0.0148]
       2007   2007    -0.0261       0.0173  [ -0.0738,   0.0217]
    ---
    Signif. codes: '*' confidence band does not cover 0

    P-value for pre-test of parallel trends assumption:  0.1681

    Control Group:  Never Treated,
    Anticipation Periods:  0
    Estimation Method:  Doubly Robust

The output displays each group-time average treatment effect. For
example, the row with Group 2004 and Time 2006 shows the effect for
counties first treated in 2004, measured in 2006 (two years after
treatment). The negative estimates suggest that minimum wage increases
reduced teen employment, with effects growing larger over time for the
2004 cohort.

Rows where Time < Group represent pre-treatment periods. Since
treatment hasn't occurred yet, any difference between groups in
these periods would indicate the groups were already diverging
before the policy change, which would violate parallel trends.
The pre-test p-value of 0.1681 indicates we cannot reject that
all pre-treatment effects are jointly zero, suggesting the groups
were evolving similarly before treatment.

The simultaneous confidence bands account for multiple testing across
all group-time cells. An asterisk (*) indicates the band excludes zero.

By default, ``att_gt`` uses doubly robust estimation, which combines
propensity score weighting with outcome regression. This provides
consistent estimates if either the propensity score model or the
outcome model is correctly specified.


Aggregating results
-------------------

While group-time effects are the fundamental building blocks, they can
be difficult to interpret when there are many groups and time periods.
The :func:`~moderndid.did.aggte` function aggregates these into more
interpretable summaries.

**Event study aggregation** shows how effects evolve relative to
treatment timing:

.. code-block:: python

    event_study = did.aggte(result, type="dynamic")
    print(event_study)

.. code-block:: text

    ==============================================================================
     Aggregate Treatment Effects (Event Study)
    ==============================================================================

     Call:
       aggte(MP, type='dynamic')

     Overall summary of ATT's based on event-study/dynamic aggregation:

           ATT      Std. Error     [95% Conf. Interval]
       -0.0772          0.0206     [ -0.1176,  -0.0369] *


     Dynamic Effects:

        Event time   Estimate   Std. Error   [95% Simult. Conf. Band]
                -3     0.0305       0.0151   [-0.0075,  0.0686]
                -2    -0.0006       0.0139   [-0.0355,  0.0344]
                -1    -0.0245       0.0142   [-0.0602,  0.0112]
                 0    -0.0199       0.0126   [-0.0516,  0.0118]
                 1    -0.0510       0.0181   [-0.0965, -0.0054] *
                 2    -0.1373       0.0366   [-0.2295, -0.0450] *
                 3    -0.1008       0.0343   [-0.1873, -0.0143] *

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

     Control Group: Never Treated
     Anticipation Periods: 0
     Estimation Method: Doubly Robust
    ==============================================================================

Event time 0 is the period of first treatment. Negative event times
are pre-treatment periods that serve as a placebo test. Effects
near zero support the parallel trends assumption. Positive event
times show how the treatment impact evolves after implementation.

In this example, pre-treatment effects (event times -3 to -1) are
small and not statistically significant, supporting parallel trends.
Post-treatment effects grow more negative over time, suggesting the
employment reduction persists and deepens after the policy change.

**Simple aggregation** produces a single overall ATT:

.. code-block:: python

    simple = did.aggte(result, type="simple")
    print(simple)

.. code-block:: text

    ==============================================================================
     Aggregate Treatment Effects
    ==============================================================================

     Call:
       aggte(MP, type='simple')

     Overall ATT:

           ATT      Std. Error     [95% Conf. Interval]
       -0.0400          0.0126     [ -0.0646,  -0.0153] *

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

     Control Group: Never Treated
     Anticipation Periods: 0
     Estimation Method: Doubly Robust
    ==============================================================================

This weighted average across all post-treatment group-time cells
gives a single summary measure. The overall ATT of -0.04 indicates
that minimum wage increases reduced log teen employment by about 4
percentage points on average.

**Group aggregation** shows the average effect for each treatment
cohort:

.. code-block:: python

    by_group = did.aggte(result, type="group")
    print(by_group)

.. code-block:: text

    ==============================================================================
     Aggregate Treatment Effects (Group/Cohort)
    ==============================================================================

     Call:
       aggte(MP, type='group')

     Overall summary of ATT's based on group/cohort aggregation:

           ATT      Std. Error     [95% Conf. Interval]
       -0.0313          0.0092     [ -0.0493,  -0.0134] *


     Group Effects:

             Group   Estimate   Std. Error   [95% Simult. Conf. Band]
              2004    -0.0797       0.0267   [-0.1422, -0.0173] *
              2006    -0.0162       0.0128   [-0.0460,  0.0136]
              2007    -0.0286       0.0103   [-0.0526, -0.0045] *

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

     Control Group: Never Treated
     Anticipation Periods: 0
     Estimation Method: Doubly Robust
    ==============================================================================

Group aggregation reveals treatment effect heterogeneity across
cohorts. Here, the 2004 cohort experienced the largest effect
(-0.08), while later adopters show smaller effects. This pattern
could reflect differences in how the policy was implemented or
in the characteristics of early versus late adopters.

**Calendar time aggregation** shows the average effect in each
calendar period:

.. code-block:: python

    by_time = did.aggte(result, type="calendar")
    print(by_time)

.. code-block:: text

    ==============================================================================
     Aggregate Treatment Effects (Calendar Time)
    ==============================================================================

     Call:
       aggte(MP, type='calendar')

     Overall summary of ATT's based on calendar time aggregation:

           ATT      Std. Error     [95% Conf. Interval]
       -0.0417          0.0168     [ -0.0746,  -0.0088] *


     Time Effects:

              Time   Estimate   Std. Error   [95% Simult. Conf. Band]
              2004    -0.0105       0.0252   [-0.0691,  0.0481]
              2005    -0.0704       0.0320   [-0.1449,  0.0041]
              2006    -0.0488       0.0212   [-0.0982,  0.0006]
              2007    -0.0371       0.0146   [-0.0709, -0.0032] *

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

     Control Group: Never Treated
     Anticipation Periods: 0
     Estimation Method: Doubly Robust
    ==============================================================================

Calendar time aggregation shows how the average treatment effect
across all treated units evolves over calendar time. This is useful
for understanding the aggregate impact of a staggered policy rollout
at each point in time.


Plotting results
----------------

ModernDiD provides built-in plotting functions using plotnine with the ``ggplot`` object.

Plot group-time effects:

.. code-block:: python

    did.plot_gt(result)

.. image:: /_static/images/plot_gt.png
   :alt: Group-time average treatment effects plot
   :width: 650px
   :align: center

Plot the event study:

.. code-block:: python

    did.plot_event_study(event_study)

.. image:: /_static/images/plot_event_study.png
   :alt: Event study plot
   :width: 650px
   :align: center


Estimation options
==================

The ``att_gt`` function provides several options for customizing the
estimation procedure. The defaults are chosen to reflect best practices,
but you may want to adjust them based on your data and research design.


Methods
-------

The ``est_method`` parameter controls how group-time effects are
estimated. The default is ``"dr"`` (doubly robust), but you can also
use inverse probability weighting or outcome regression:

.. code-block:: python

    # Doubly robust (default) - combines IPW and regression
    result_dr = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="dr",
    )

    # Inverse probability weighting
    result_ipw = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="ipw",
    )

    # Outcome regression
    result_reg = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        est_method="reg",
    )

Doubly robust estimation is recommended because it provides consistent
estimates if either the propensity score model or the outcome regression
model is correctly specified.


Adding covariates
-----------------

When treated and comparison groups differ on observable characteristics,
you can include covariates using the ``xformla`` parameter. This
strengthens the parallel trends assumption by conditioning on these
differences:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        xformla="~ lpop",
    )

The formula uses R-style syntax. Multiple covariates can be included
as ``xformla="~ x1 + x2 + x3"``. Covariates should be time-invariant
or measured before treatment to avoid conditioning on post-treatment
variables.


Control group options
---------------------

By default, ``att_gt`` uses never-treated units as the comparison
group. You can instead use not-yet-treated units, which includes
units that will be treated in the future but have not yet been
treated at time *t*:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        control_group="notyettreated",
    )

Using not-yet-treated units can increase precision when there are
few never-treated units, but requires assuming that treatment timing
is unrelated to potential outcomes. If early adopters differ
systematically from late adopters, this assumption may be violated.


Bootstrap inference
-------------------

By default, ``att_gt`` uses the multiplier bootstrap to compute
standard errors and simultaneous confidence bands. You can control
the bootstrap settings:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        bstrap=True,       # Use bootstrap (default)
        biters=1000,       # Number of bootstrap iterations (default)
        cband=True,        # Compute uniform confidence bands (default)
        alp=0.05,          # Significance level (default)
        random_state=42,   # For reproducibility
    )

Setting ``bstrap=False`` uses analytical standard errors instead of
bootstrap, which is faster but does not support clustering or uniform
confidence bands. Setting ``cband=False`` computes pointwise confidence
intervals instead of simultaneous bands.


Clustering
----------

Standard errors can be clustered at one or two levels using the
``clustervars`` parameter. Clustering requires using the bootstrap:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        clustervars=["countyreal"],  # Cluster at unit level
        bstrap=True,
    )

You can cluster on at most two variables, and one of them must be the
unit identifier (``idname``).


Anticipation effects
--------------------

If units can anticipate treatment before it actually occurs, you can
account for this using the ``anticipation`` parameter. This shifts the
base period backward:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        anticipation=1,  # Allow 1 period of anticipation
    )

With ``anticipation=1``, the base period for group *g* becomes period
*g-2* instead of *g-1*, treating period *g-1* as potentially affected
by anticipation of the upcoming treatment.


Base period
-----------

The ``base_period`` parameter controls which pre-treatment period is
used as the comparison. The default is ``"varying"``, which uses the
period immediately before treatment (or before anticipation). Setting
``base_period="universal"`` uses the same base period for all groups:

.. code-block:: python

    # Varying base period (default)
    result_varying = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        base_period="varying",
    )

    # Universal base period
    result_universal = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        base_period="universal",
    )

A universal base period can be useful when you want all pre-treatment
estimates to be relative to the same reference point.


Panel data options
------------------

By default, ``att_gt`` expects a balanced panel where all units are
observed in all time periods. If your panel is unbalanced, you can
set ``allow_unbalanced_panel=True``:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        allow_unbalanced_panel=True,
    )

For repeated cross-sections rather than panel data, set ``panel=False``
and omit the ``idname`` parameter.


Sampling weights
----------------

If your data has sampling weights, specify them using ``weightsname``:

.. code-block:: python

    result = did.att_gt(
        data=data,
        yname="lemp",
        tname="year",
        idname="countyreal",
        gname="first.treat",
        weightsname="weight_column",
    )


Next steps
==========

This quickstart covered the basic workflow for staggered DiD estimation.
ModernDiD provides additional capabilities including:

- **Doubly robust two-period DiD** (``drdid``) for classic two-period
  settings with a single pre and post period.
- **Continuous treatment DiD** (:mod:`~moderndid.didcont`) for settings
  with treatment intensity. Adds ``dname`` for dose and produces
  dose-response functions.
- **Triple difference-in-differences** (:mod:`~moderndid.didtriple`) for
  designs with an additional dimension of variation. Adds ``ename`` for
  the eligibility indicator.
- **Sensitivity analysis** (:mod:`~moderndid.didhonest`) for assessing
  robustness to parallel trends violations. Takes results from ``att_gt``
  and produces robust confidence intervals.

See the user guide for detailed coverage of these methods.
