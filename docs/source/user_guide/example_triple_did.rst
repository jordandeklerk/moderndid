.. _example_triple_did:

=======================================
Triple Difference-in-Differences
=======================================

Some policies create a natural within-group comparison. A parental leave
mandate affects women but not men. A minimum wage increase hits hourly workers
but not salaried employees. An education reform applies to public schools but
not private ones.

Triple difference-in-differences (DDD) exploits this structure. Beyond comparing
treated and control groups before and after policy change, it adds a third
comparison between eligible and ineligible subgroups within each group. This
additional dimension of variation strengthens causal identification when
parallel trends across groups alone might not hold.

The `Ortiz-Villavicencio and Sant'Anna (2025) <https://arxiv.org/abs/2505.09942>`_
estimator provides doubly robust inference for DDD designs, supporting both
two-period and staggered adoption settings.


Three dimensions of variation
------------------------------

DDD exploits variation along three dimensions simultaneously.

**Treatment status** distinguishes groups where treatment is enabled (such as
states that pass a policy) from control groups (states that do not). In the
data, this is encoded in the group variable ``gname``.

**Eligibility** distinguishes units within groups who are affected by the
policy from those who are not. A parental leave policy affects women but not
men. A minimum wage increase affects hourly workers but not salaried employees.
This partition is encoded in the ``pname`` variable, where 1 indicates eligible
units and 0 indicates ineligible units.

**Time** distinguishes pre-treatment from post-treatment periods, as in
standard DiD. This creates four subgroups whose outcome changes we can compare.

- Treated and eligible units receive the actual treatment
- Treated but ineligible units are in treated groups but unaffected
- Eligible but untreated units would be affected if their group were treated
- Untreated and ineligible units are controls on both dimensions

The DDD estimator computes treatment effects by differencing out trends that
are common across groups or across eligibility status, isolating the effect
of treatment on eligible units in treated groups.


Assumptions of DDD
-------------------

DDD relaxes the standard DiD assumption in an important way. Standard DiD
requires that treated and control groups would have followed parallel trends
absent treatment. DDD instead requires that the *gap* between eligible and
ineligible units evolves similarly across treated and control groups.

This is weaker because DDD allows for the following.

- Different trends between treated and control groups (as long as eligible
  and ineligible units within each group diverge similarly)
- Different trends between eligible and ineligible units (as long as this
  difference is stable across groups)

DDD is appropriate when you have reason to believe that group-level or
eligibility-level confounders exist, but that these confounders affect
eligible and ineligible units similarly within groups.


Common pitfalls of DDD
-----------------------

There are several common pitfalls to avoid when estimating DDD.

- **Computing DDD as the difference of two DiDs.** Running DiD separately
  within treated groups (eligible vs ineligible) and control groups, then
  subtracting, produces biased estimates when covariates matter. Each DiD
  integrates over its own covariate distribution rather than the treated
  population's distribution.

- **Pooling not-yet-treated groups in staggered designs.** Unlike standard
  staggered DiD, pooling all not-yet-treated units as a single comparison
  group can bias DDD estimates. Different cohorts may have different
  eligibility compositions, and the DDD assumption allows group-specific
  trend deviations that don't average out when pooling. The estimator here
  uses each comparison cohort separately and combines them optimally.

- **Relying on three-way fixed effects with covariates.** Adding covariates
  linearly to a three-way fixed effects regression doesn't properly account
  for covariate-specific trends in DDD designs. Use the doubly robust
  estimator instead, which correctly integrates over the treated units'
  covariate distribution.


Simulating data
---------------

For this walkthrough, we use simulated panel data with the four subgroups
described above.

.. code-block:: python

    import moderndid as did

    dgp = did.gen_dgp_2periods(n=5000, dgp_type=1, random_state=42)
    data = dgp["data"]
    print(data.head(6))

.. code-block:: text

    shape: (6, 10)
    ┌─────┬───────┬───────────┬──────┬───┬───────────┬───────────┬──────────┬─────────┐
    │ id  ┆ state ┆ partition ┆ time ┆ … ┆ cov2      ┆ cov3      ┆ cov4     ┆ cluster │
    │ --- ┆ ---   ┆ ---       ┆ ---  ┆   ┆ ---       ┆ ---       ┆ ---      ┆ ---     │
    │ i64 ┆ i64   ┆ i64       ┆ i64  ┆   ┆ f64       ┆ f64       ┆ f64      ┆ i64     │
    ╞═════╪═══════╪═══════════╪══════╪═══╪═══════════╪═══════════╪══════════╪═════════╡
    │ 1   ┆ 0     ┆ 1         ┆ 1    ┆ … ┆ -0.20212  ┆ -0.012154 ┆ 0.862487 ┆ 12      │
    │ 1   ┆ 0     ┆ 1         ┆ 2    ┆ … ┆ -0.20212  ┆ -0.012154 ┆ 0.862487 ┆ 12      │
    │ 2   ┆ 1     ┆ 1         ┆ 1    ┆ … ┆ -0.408526 ┆ -0.916222 ┆ -0.09073 ┆ 5       │
    │ 2   ┆ 1     ┆ 1         ┆ 2    ┆ … ┆ -0.408526 ┆ -0.916222 ┆ -0.09073 ┆ 5       │
    │ 3   ┆ 0     ┆ 1         ┆ 1    ┆ … ┆ -0.412515 ┆ -1.059306 ┆ 1.064033 ┆ 1       │
    │ 3   ┆ 0     ┆ 1         ┆ 2    ┆ … ┆ -0.412515 ┆ -1.059306 ┆ 1.064033 ┆ 1       │
    └─────┴───────┴───────────┴──────┴───┴───────────┴───────────┴──────────┴─────────┘

The data has 5,000 units observed across 2 time periods. The ``state`` variable
indicates treatment group membership (0 for control, 1 for treated). The
``partition`` variable indicates eligibility (0 for ineligible, 1 for eligible).
This simulation has no true treatment effect, so the DDD estimate should be
close to zero.


Two-period DDD estimation
-------------------------

With a simple two-period design where all treated units receive treatment at
the same time, the ``ddd`` function estimates the average treatment effect on
treated eligible units.

.. code-block:: python

    result = did.ddd(
        data=data,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
    )
    print(result)

.. code-block:: text

    ==============================================================================
     Triple Difference-in-Differences (DDD) Estimation
    ==============================================================================

     DR-DDD estimation for the ATT:

    ┌────────┬────────────┬──────────┬────────────────────────┐
    │    ATT │ Std. Error │ Pr(>|t|) │ [95% Conf. Interval]   │
    ├────────┼────────────┼──────────┼────────────────────────┤
    │ 0.0229 │     0.0828 │   0.7825 │ [ -0.1394,   0.1851]   │
    └────────┴────────────┴──────────┴────────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence interval does not cover 0

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------
     Panel Data: 2 periods
     Outcome variable: y
     Qualification variable: partition

     No. of units at each subgroup:
       treated-and-eligible: 1235
       treated-but-ineligible: 1246
       eligible-but-untreated: 1291
       untreated-and-ineligible: 1228

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------
     Outcome regression: OLS
     Propensity score: Logistic regression (MLE)

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Analytical standard errors
    ==============================================================================
     See Ortiz-Villavicencio and Sant'Anna (2025) for details.

We get an ATT estimate of 0.023, close to zero as expected, with a confidence
interval that comfortably includes zero. The output also reports the number of
units in each of the four subgroups, confirming balanced representation across
the design.

The ``est_method="dr"`` option uses doubly robust estimation, which combines
outcome regression and propensity score weighting. This approach is consistent
if either the outcome model or the propensity score model is correctly
specified, providing robustness against model misspecification.


Staggered treatment adoption
----------------------------

When treatment adoption is staggered across groups (some groups adopt in period
2, others in period 3, etc.), we can estimate group-time average treatment
effects for each combination of treatment cohort and time period.

.. code-block:: python

    dgp_mp = did.gen_dgp_mult_periods(n=500, dgp_type=1, random_state=42)
    data_mp = dgp_mp["data"]
    print(data_mp.head(6))

.. code-block:: text

    shape: (6, 10)
    ┌─────┬───────┬───────────┬──────┬───┬──────────┬───────────┬───────────┬─────────┐
    │ id  ┆ group ┆ partition ┆ time ┆ … ┆ cov2     ┆ cov3      ┆ cov4      ┆ cluster │
    │ --- ┆ ---   ┆ ---       ┆ ---  ┆   ┆ ---      ┆ ---       ┆ ---       ┆ ---     │
    │ i64 ┆ i64   ┆ i64       ┆ i64  ┆   ┆ f64      ┆ f64       ┆ f64       ┆ i64     │
    ╞═════╪═══════╪═══════════╪══════╪═══╪══════════╪═══════════╪═══════════╪═════════╡
    │ 1   ┆ 2     ┆ 1         ┆ 1    ┆ … ┆ 1.068661 ┆ -0.081955 ┆ -0.218837 ┆ 14      │
    │ 1   ┆ 2     ┆ 1         ┆ 2    ┆ … ┆ 1.068661 ┆ -0.081955 ┆ -0.218837 ┆ 14      │
    │ 1   ┆ 2     ┆ 1         ┆ 3    ┆ … ┆ 1.068661 ┆ -0.081955 ┆ -0.218837 ┆ 14      │
    │ 2   ┆ 3     ┆ 1         ┆ 1    ┆ … ┆ 1.221115 ┆ 0.709174  ┆ -1.161969 ┆ 17      │
    │ 2   ┆ 3     ┆ 1         ┆ 2    ┆ … ┆ 1.221115 ┆ 0.709174  ┆ -1.161969 ┆ 17      │
    │ 2   ┆ 3     ┆ 1         ┆ 3    ┆ … ┆ 1.221115 ┆ 0.709174  ┆ -1.161969 ┆ 17      │
    └─────┴───────┴───────────┴──────┴───┴──────────┴───────────┴───────────┴─────────┘

In multi-period data, the ``group`` variable indicates when treatment is first
enabled for each unit (0 for never-treated, 2 for treated starting in period 2,
etc.). This simulation has positive treatment effects that grow over time.

.. code-block:: python

    result_mp = did.ddd(
        data=data_mp,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        control_group="nevertreated",
        base_period="universal",
        est_method="dr",
    )
    print(result_mp)

.. code-block:: text

    ==============================================================================
     Triple Difference-in-Differences (DDD) Estimation
     Multi-Period / Staggered Treatment Adoption
    ==============================================================================

     DR-DDD estimation for ATT(g,t):

    ┌───────┬──────┬──────────┬────────────┬──────────────────────┐
    │ Group │ Time │ ATT(g,t) │ Std. Error │ [95% Conf. Int.]     │
    ├───────┼──────┼──────────┼────────────┼──────────────────────┤
    │     2 │    1 │   0.0000 │         NA │ NA                   │
    │     2 │    2 │  11.1769 │     0.4201 │ [10.3535, 12.0004] * │
    │     2 │    3 │  21.1660 │     0.4516 │ [20.2808, 22.0511] * │
    │     3 │    1 │  -1.0095 │     0.5450 │ [-2.0778,  0.0587]   │
    │     3 │    2 │   0.0000 │         NA │ NA                   │
    │     3 │    3 │  24.9440 │     0.4724 │ [24.0182, 25.8698] * │
    └───────┴──────┴──────────┴────────────┴──────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence interval does not cover 0

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------
     Panel Data
     Outcome variable: y
     Qualification variable: partition
     Control group: Never Treated
     Base period: universal

     No. of units per treatment group:
       Units never enabling treatment: 97
       Units enabling treatment at period 2: 173
       Units enabling treatment at period 3: 230

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------
     Outcome regression: OLS
     Propensity score: Logistic regression (MLE)

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Analytical standard errors
    ==============================================================================
     See Ortiz-Villavicencio and Sant'Anna (2025) for details.

Each row shows the ATT for a specific cohort at a specific time. Rows where
Time is before the Group's treatment date (like Group 2, Time 1) serve as
placebo tests. These pre-treatment estimates should be close to zero if the
DDD parallel trends assumption is plausible. The estimate of -1.0095 for Group 3 at
Time 1 is small relative to the post-treatment effects and statistically
insignificant, consistent with the DDD parallel trends assumption.

The post-treatment effects are large and precisely estimated. Group 2 shows
effects of 11.2 at time 2 growing to 21.2 at time 3, while Group 3 shows an
effect of 24.9 at time 3. The growth from period to period for Group 2
suggests that treatment effects accumulate rather than appearing all at once.
Group 3's larger single-period effect (24.9 vs 11.2 for Group 2 at on-impact)
reflects the different treatment effect magnitudes built into the simulation.
In applied work, such heterogeneity could arise from differences in cohort
composition or treatment intensity across groups.


Aggregating into an event study
-------------------------------

With multiple group-time estimates, there is a lot to digest. The event study
aggregation aligns all cohorts relative to their treatment date, making it
much easier to see the overall pattern.

.. code-block:: python

    event_study = did.agg_ddd(result_mp, aggregation_type="eventstudy")
    print(event_study)

.. code-block:: text

    ==============================================================================
     Aggregate DDD Treatment Effects (Event Study)
    ==============================================================================

     Overall summary of ATT's based on event-study aggregation:

    ┌─────────┬────────────┬────────────────────────┐
    │     ATT │ Std. Error │ [95% Conf. Interval]   │
    ├─────────┼────────────┼────────────────────────┤
    │ 20.1000 │     0.3567 │ [ 19.4010,  20.7990] * │
    └─────────┴────────────┴────────────────────────┘


     Dynamic Effects:

    ┌────────────┬──────────┬────────────┬──────────────────────────┐
    │ Event time │ Estimate │ Std. Error │ [95% Simult. Conf. Band] │
    ├────────────┼──────────┼────────────┼──────────────────────────┤
    │         -2 │  -1.0095 │     0.5436 │ [-2.2865,  0.2675]       │
    │         -1 │   0.0000 │         NA │ NA                       │
    │          0 │  19.0341 │     0.2571 │ [18.4301, 19.6380] *     │
    │          1 │  21.1660 │     0.4517 │ [20.1048, 22.2271] *     │
    └────────────┴──────────┴────────────┴──────────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Bootstrap standard errors
    ==============================================================================
     See Ortiz-Villavicencio and Sant'Anna (2025) for details.

Event time 0 is the period of treatment adoption. The pre-treatment estimate
at event time -2 is -1.0 and statistically insignificant, consistent
with the parallel trends assumption. The on-impact effect at event time 0 is 19.0,
growing to 21.2 by event time 1.


Summarizing by cohort
---------------------

We can also look at average effects for each treatment cohort, revealing
heterogeneity across groups that adopted at different times.

.. code-block:: python

    group_agg = did.agg_ddd(result_mp, aggregation_type="group")
    print(group_agg)

.. code-block:: text

    ==============================================================================
     Aggregate DDD Treatment Effects (Group/Cohort)
    ==============================================================================

     Overall summary of ATT's based on group/cohort aggregation:

    ┌─────────┬────────────┬────────────────────────┐
    │     ATT │ Std. Error │ [95% Conf. Interval]   │
    ├─────────┼────────────┼────────────────────────┤
    │ 21.1781 │     0.3947 │ [ 20.4045,  21.9517] * │
    └─────────┴────────────┴────────────────────────┘


     Group Effects:

    ┌───────┬──────────┬────────────┬──────────────────────────┐
    │ Group │ Estimate │ Std. Error │ [95% Simult. Conf. Band] │
    ├───────┼──────────┼────────────┼──────────────────────────┤
    │     2 │  16.1715 │     0.3584 │ [15.3514, 16.9915] *     │
    │     3 │  24.9440 │     0.5008 │ [23.7980, 26.0899] *     │
    └───────┴──────────┴────────────┴──────────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Bootstrap standard errors
    ==============================================================================
     See Ortiz-Villavicencio and Sant'Anna (2025) for details.

Group 2 (early adopters) shows an average effect of 16.2, while Group 3 (later
adopters) shows a larger effect of 24.9. This heterogeneity could reflect
different treatment intensities, composition effects, or the fact that later
cohorts have fewer post-treatment periods averaged into their estimates.


Overall average effect
----------------------

If you need a single summary number, the simple aggregation averages across
all post-treatment group-time cells.

.. code-block:: python

    simple_agg = did.agg_ddd(result_mp, aggregation_type="simple")
    print(simple_agg)

.. code-block:: text

    ==============================================================================
     Aggregate DDD Treatment Effects
    ==============================================================================

     Overall ATT:

    ┌─────────┬────────────┬────────────────────────┐
    │     ATT │ Std. Error │ [95% Conf. Interval]   │
    ├─────────┼────────────┼────────────────────────┤
    │ 19.6744 │     0.2917 │ [ 19.1027,  20.2460] * │
    └─────────┴────────────┴────────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence band does not cover 0

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Bootstrap standard errors
    ==============================================================================
     See Ortiz-Villavicencio and Sant'Anna (2025) for details.

The overall ATT of 19.67 represents the average treatment effect across all
treated eligible units in all post-treatment periods, weighted by group size.


Plotting results
----------------

Visualizations make it easier to communicate findings. We can start with the
group-time plot showing all underlying estimates organized by cohort.

.. code-block:: python

    did.plot_gt(result_mp)

.. image:: /_static/images/plot_ddd_gt.png
   :alt: Group-time DDD treatment effects plot
   :width: 100%

The event study plot shows effects relative to treatment adoption with
confidence bands. The vertical dotted line marks the reference period.

.. code-block:: python

    did.plot_event_study(event_study, ref_period=-1)

.. image:: /_static/images/plot_ddd_event_study.png
   :alt: DDD event study plot
   :width: 100%

The flat pre-treatment estimate is consistent with the identifying assumptions,
though passing pre-treatment placebos alone does not guarantee that the
assumption holds in post-treatment periods. The growing post-treatment effects
indicate a positive and accumulating treatment effect.


Control group options
---------------------

Like the standard staggered DiD estimator, DDD supports different control
group definitions.

**Never Treated** (default) uses only units that never receive treatment
during the sample period as controls.

**Not Yet Treated** uses units that will eventually be treated but have not
yet received treatment as additional controls. This can improve efficiency
but requires stronger assumptions.

.. code-block:: python

    result_nyt = did.ddd(
        data=data_mp,
        yname="y",
        tname="time",
        idname="id",
        gname="group",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        control_group="notyettreated",
        base_period="universal",
        est_method="dr",
    )

With not-yet-treated controls, some standard errors may be smaller because
more observations contribute to the control group comparison.


Repeated cross-section data
---------------------------

If your data is a repeated cross-section where different units are sampled in
each period, that works too. Just set ``panel=False``.

.. code-block:: python

    dgp_rcs = did.gen_dgp_2periods(n=5000, dgp_type=1, panel=False, random_state=42)
    data_rcs = dgp_rcs["data"]

    result_rcs = did.ddd(
        data=data_rcs,
        yname="y",
        tname="time",
        idname="id",
        gname="state",
        pname="partition",
        xformla="~ cov1 + cov2 + cov3 + cov4",
        est_method="dr",
        panel=False,
    )

The estimation method automatically adapts to the repeated cross-section
structure, fitting separate outcome models for each subgroup rather than
tracking individual units over time.


Next steps
----------

For details on additional estimation options including bootstrap inference,
clustering, and estimation methods, see the
:ref:`Triple DiD API reference <api-didtriple>`.

For theoretical background on triple difference-in-differences, see the
:ref:`Background <background-tripledid>` section.

For the standard two-difference version that this estimator extends, see the
:ref:`Staggered DiD walkthrough <example_staggered_did>`.
