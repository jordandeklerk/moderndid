.. _example_staggered_did:

=======================================
Staggered Difference-in-Differences
=======================================

Did minimum wage increases reduce teen employment? Economists have long
studied this question using difference-in-differences designs that compare
employment trends in states that raised their minimum wage to states that
did not.

When states adopt policies at different times, standard two-way fixed effects
regression can produce misleading results. The
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_
estimator addresses this by computing separate treatment effects for each
cohort and time period, then aggregating them into interpretable summaries
like event studies.

The following example walks through the complete workflow using county-level
employment data from states that raised their minimum wage between 2004 and
2007. For API details, see the :ref:`quickstart <quickstart>`. For conceptual
background, see the :ref:`Introduction to Difference-in-Differences <causal_inference>`.


Loading data
------------

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

The dataset contains 500 counties observed annually from 2003 to 2007. Some
counties are in states that raised their minimum wage in 2004, others in 2006
or 2007, and some never raised it during this period. The ``first.treat``
variable encodes this timing, with 0 indicating never-treated counties.

This is a balanced panel where each county appears exactly once per year.
The estimator can handle unbalanced panels, but balanced data simplifies
interpretation and improves precision.


Estimating group-time effects
-----------------------------

The first step is to estimate treatment effects separately for each cohort
at each time period. This avoids the negative weighting problem that can
bias two-way fixed effects estimates when treatment effects vary across
cohorts or over time.

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

Each row represents a specific cohort (``gname``) measured at a specific time
(``tname``). Cohorts are defined by when they first received treatment. For
the 2004 cohort, the rows for 2004-2007 show how the treatment effect
evolves from the year of adoption through three years later.

Rows where Time is less than Group are pre-treatment periods. These should
be close to zero if parallel trends holds. For the 2007 cohort, the estimates
at times 2004, 2005, and 2006 are all pre-treatment. The estimate of 0.0305
in 2004 is slightly positive but not statistically significant, and the joint
pre-test p-value of 0.1681 suggests we cannot reject parallel trends at
conventional levels. A low p-value here would be a warning sign that the
identifying assumption may be violated.

For the 2004 cohort, effects grow from -0.01 in 2004 to -0.14 in 2006 before
moderating to -0.10 in 2007. This pattern of growing then stabilizing effects
is common and suggests the policy had a lagged impact that took time to fully
materialize. The 2006 and 2007 cohorts show smaller and less precisely
estimated effects, possibly because they have fewer post-treatment periods
to observe.


Aggregating into an event study
-------------------------------

With 12 group-time estimates, the results are difficult to summarize. The
event study aggregation aligns all cohorts relative to their treatment date,
making it easier to see the overall pattern of effects over time.

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

Event time 0 is the year of treatment adoption. Negative event times are
pre-treatment periods that serve as a placebo test for parallel trends.
Positive event times show how effects evolve after treatment.

The estimates at event times -3, -2, and -1 are all close to zero and
statistically insignificant. This is reassuring. If these were large or
showed a trending pattern, it would suggest treated and control counties
were already diverging before the policy change, calling into question
whether the post-treatment differences reflect the policy or pre-existing
trends.

The effect is small and insignificant on impact (event time 0), but grows
to -0.05 after one year and -0.14 after two years. This lag makes economic
sense. Employers may not immediately respond to a minimum wage increase.
They may first absorb higher costs, then gradually reduce hours or hiring
as contracts expire and business conditions allow adjustments. The effect
moderates slightly at event time 3, though this estimate is based on fewer
cohorts and is less precisely estimated.


Summarizing the overall effect
------------------------------

For policy discussions, you often need a single number that summarizes the
average treatment effect. The ``"simple"`` aggregation provides this by
averaging across all post-treatment group-time cells, weighted by group size.

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

The overall ATT of -0.04 means that minimum wage increases reduced log teen
employment by about 4 log points (approximately 4 percent) on average across
all treated counties and post-treatment periods. The confidence interval
excludes zero, indicating this effect is statistically significant.

This single number is useful for policy summaries but masks the treatment
effect heterogeneity we saw in the event study. It is generally a good idea to
examine the dynamic effects before reporting only the overall ATT.


Examining heterogeneity by cohort
---------------------------------

Different cohorts may experience different effects due to variation in
local economic conditions, policy implementation, or the composition of
affected workers. The ``"group"`` aggregation reveals this heterogeneity.

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
       -0.0310          0.0122     [ -0.0550,  -0.0071] *


     Group Effects:

             Group   Estimate   Std. Error   [95% Simult. Conf. Band]
              2004    -0.0797       0.0279   [-0.1430, -0.0165] *
              2006    -0.0229       0.0173   [-0.0620,  0.0162]
              2007    -0.0261       0.0175   [-0.0656,  0.0135]

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

     Control Group: Never Treated
     Anticipation Periods: 0
     Estimation Method: Doubly Robust
    ==============================================================================

The 2004 cohort experienced the largest effect at -0.08, while the 2006 and
2007 cohorts show effects around -0.02 to -0.03 that are not statistically
significant. This heterogeneity could arise for several reasons. Early
adopters may have implemented larger minimum wage increases. They also have
more post-treatment periods in the data, so their effects have had more time
to materialize. Or the economic conditions in 2004 may have made employment
more sensitive to wage floors than in later years.

When you see substantial heterogeneity across cohorts, the burden is on the researcher to
consider whether it makes sense to report a single overall effect or whether the story is more
nuanced. It is generally a good idea to report the event study and the cohort-specific effects
to provide a more complete picture.


Plotting results
----------------

Visualizations make it easier to communicate findings and spot patterns.
The group-time plot shows all the underlying estimates organized by cohort.

.. code-block:: python

    did.plot_gt(result)

.. image:: /_static/images/plot_gt.png
   :alt: Group-time average treatment effects plot
   :width: 100%

The event study plot is typically the most informative visualization for
staggered designs. It clearly shows both the pre-trend test and the
post-treatment dynamics.

.. code-block:: python

    did.plot_event_study(event_study)

.. image:: /_static/images/plot_event_study.png
   :alt: Event study plot
   :width: 100%

The flat pre-treatment estimates and the growing post-treatment effects
tell a clear story. This is the kind of pattern that builds confidence
in a causal interpretation.


Next steps
----------

For details on estimation options such as covariates, control groups,
bootstrap inference, and clustering, see the :ref:`quickstart <quickstart>` or
the :ref:`API reference <api-multiperiod>`.

For theoretical background on the Callaway and Sant'Anna estimator, see
the :ref:`Background <background>` section.
