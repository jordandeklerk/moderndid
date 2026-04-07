.. _example_dyn_balancing:

========================================
Dynamic Covariate Balancing DiD
========================================

When treatments change over time and units dynamically select into
treatment based on past outcomes, standard DiD methods break down. The
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_
estimator handles this by combining potential local projections with
sequential covariate balancing weights, producing valid inference without
requiring propensity score estimation.

.. seealso::

   :ref:`Dynamic Covariate Balancing DiD <background-diddynamic>` for the
   theoretical foundations, identifying assumptions, and the dynamic
   balancing algorithm.


Empirical application
---------------------

This example follows the empirical analysis in
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_,
which revisits `Acemoglu et al. (2019) <https://doi.org/10.1086/700936>`_
on the effect of democracy on economic growth.

The dataset contains 141 countries observed across six five-year periods
from 1989 to 2010. The outcome is log GDP per capita and the treatment is
a binary democracy indicator. Unlike standard DiD settings where
treatment is permanent, democracy can switch on and off across periods.
About two-thirds of country-period observations are democratic, but many
countries transition between democracy and autocracy multiple times. This
switching creates dynamic selection into treatment, where a country's
current democratic status depends on its past economic performance and
governance history, violating the staggered adoption assumption required
by standard DiD.

The dataset includes 158 country-level covariates, 4 lagged outcome
variables, and a geographic region indicator.
`Acemoglu et al. (2019) <https://doi.org/10.1086/700936>`_ assume a
dynamic selection model where treatment decisions depend on past outcomes
and covariates, making this an ideal application for DCB.
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_
show that standard local projections and the linear regression approach
of `Acemoglu et al. (2019) <https://doi.org/10.1086/700936>`_
substantially underestimate the long-run effect of democracy, while IPW
methods produce unstable estimates due to the compounding of propensity
scores over multiple periods.


Loading the data
^^^^^^^^^^^^^^^^

.. code-block:: python

    from moderndid.core.data import load_acemoglu
    import polars as pl

    df = load_acemoglu()

    # The estimator requires numeric unit identifiers
    units = sorted(df["Unit"].unique().to_list())
    unit_map = {u: i for i, u in enumerate(units)}
    df = df.with_columns(pl.col("Unit").replace(unit_map).cast(pl.Int64))

    print(df.select("Y", "D", "Unit", "Time", "region", "V1", "V2").head(6))

.. code-block:: text

    shape: (6, 7)
    ┌──────────┬─────┬──────┬──────┬────────┬───────────┬───────────┐
    │ Y        ┆ D   ┆ Unit ┆ Time ┆ region ┆ V1        ┆ V2        │
    │ ---      ┆ --- ┆ ---  ┆ ---  ┆ ---    ┆ ---       ┆ ---       │
    │ f64      ┆ i64 ┆ i64  ┆ i64  ┆ str    ┆ f64       ┆ f64       │
    ╞══════════╪═════╪══════╪══════╪════════╪═══════════╪═══════════╡
    │ 7.330224 ┆ 1   ┆ 70   ┆ 0    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    │ 7.374312 ┆ 1   ┆ 70   ┆ 1    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    │ 7.427509 ┆ 1   ┆ 70   ┆ 2    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    │ 7.497994 ┆ 1   ┆ 70   ┆ 3    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    │ 7.526908 ┆ 1   ┆ 70   ┆ 4    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    │ 7.557471 ┆ 1   ┆ 70   ┆ 5    ┆ SA     ┆ -0.398948 ┆ -0.064505 │
    └──────────┴─────┴──────┴──────┴────────┴───────────┴───────────┘

The ``Y`` column is log GDP per capita. The ``D`` column is the
democracy indicator, which can switch on and off across periods. The
six time periods (0--5) correspond to five-year intervals from 1989 to
2010. The dataset also includes 158 country-level covariates
(``V1``--``V158``), 4 lagged outcome variables
(``lag1.Value1``--``lag4.Value1``), and a ``region`` variable encoding
the World Bank geographic region.


Estimation
^^^^^^^^^^

We start by estimating the average treatment effect of being democratic
for two consecutive periods versus not being democratic. The treatment
histories ``ds1=[1, 1]`` and ``ds2=[0, 0]`` specify these two sequences,
read left to right from the earliest to the most recent period.

.. code-block:: python

    from moderndid.diddynamic import dyn_balancing

    result = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ V1 + V2 + V3 + V4 + V5",
        fixed_effects=["region"],
    )
    print(result)

.. code-block:: text

    ==============================================================================
     Dynamic Covariate Balancing Estimation
    ==============================================================================

     DCB estimation for the ATE:

    ┌────────┬────────────┬──────────┬────────────────────────┐
    │    ATE │ Std. Error │ Pr(>|t|) │ [95% Conf. Interval]   │
    ├────────┼────────────┼──────────┼────────────────────────┤
    │ 0.3011 │     0.2032 │   0.1383 │ [ -0.0971,   0.6993]   │
    └────────┴────────────┴──────────┴────────────────────────┘

    ------------------------------------------------------------------------------
     Signif. codes: '*' confidence interval does not cover 0

    ------------------------------------------------------------------------------
     Potential Outcomes
    ------------------------------------------------------------------------------
     mu(ds1):  8.0044  (0.1397)
     mu(ds2):  7.7033  (0.1476)

    ------------------------------------------------------------------------------
     Data Info
    ------------------------------------------------------------------------------
     Treatment history ds1: [1, 1]
     Treatment history ds2: [0, 0]
     Outcome variable: Y
     Units: 137
     Observations: 274

    ------------------------------------------------------------------------------
     Estimation Details
    ------------------------------------------------------------------------------
     Balancing: DCB
     Coefficient estimation: lasso_plain

    ------------------------------------------------------------------------------
     Inference
    ------------------------------------------------------------------------------
     Significance level: 0.05
     Analytical standard errors
     Robust (chi-squared) critical values
    ==============================================================================
     Viviano and Bradic (2026)

The estimated ATE of 0.30 means that two consecutive periods of
democracy increase log GDP per capita by about 0.30 (roughly 35% in
levels) compared to two periods without democracy. The potential outcome
estimates ``mu(ds1)`` and ``mu(ds2)`` show the average outcomes under
each history, with standard errors in parentheses.

Inference uses robust chi-squared critical values by default, which
provide valid coverage under weaker conditions than Gaussian quantiles
by accounting for the estimation error of the balancing weights. With a
p-value of 0.14, the effect is not statistically significant at the 5%
level with this parsimonious specification. The history length analysis
below examines how the effect varies with exposure duration.

The ``xformla`` argument specifies which covariates to include in the
LASSO regression. Here we use a small subset (``V1``--``V5``) of the
158 available covariates. The ``fixed_effects`` argument adds region
dummies to the model without penalisation. Any unit not observed in all
periods within the treatment history window is automatically dropped
(here 4 units, leaving 137).


Interpreting the result object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The result is a ``DynBalancingResult`` containing the ATE, variances,
potential outcomes, and diagnostics for the balancing procedure.

.. code-block:: python

    print(f"ATE:     {result.att:.4f}")
    print(f"SE:      {result.se:.4f}")
    print(f"mu(ds1): {result.mu1:.4f}")
    print(f"mu(ds2): {result.mu2:.4f}")
    print(f"Robust quantile:   {result.robust_quantile:.4f}")
    print(f"Gaussian quantile: {result.gaussian_quantile:.4f}")
    print(f"Imbalance ds1: {result.imbalances['ds1']:.6f}")
    print(f"Imbalance ds2: {result.imbalances['ds2']:.6f}")

.. code-block:: text

    ATE:     0.3011
    SE:      0.2032
    mu(ds1): 8.0044
    mu(ds2): 7.7033
    Robust quantile:   3.8415
    Gaussian quantile: 1.9600
    Imbalance ds1: 0.000012
    Imbalance ds2: 0.000009

The imbalance measures report the maximum covariate imbalance (rescaled
by standard deviation) after the DCB weights have been applied. Values
close to zero confirm that the balancing procedure has successfully
equalised the covariate distributions between treatment groups. This is
a key advantage of DCB over IPW, which can leave substantial residual
imbalance when propensity scores are difficult to estimate. The
``gammas`` dictionary contains the actual balancing weights per treatment
history, and ``coefficients`` contains the LASSO-estimated regression
coefficients used in the recursive local projections.


Treatment history length
------------------------

With long panels, using the full treatment history can thin the
effective sample because only units matching the entire treatment path
contribute non-zero weights. Setting ``histories_length`` traces out
how the treatment effect evolves with exposure length. Here we compare
the effect of being democratic for 1, 2, 3, 4, and 5 consecutive
periods against the corresponding non-democratic histories.

.. code-block:: python

    history = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1, 1, 1, 1],
        ds2=[0, 0, 0, 0, 0],
        histories_length=[1, 2, 3, 4, 5],
        xformla="~ V1 + V2 + V3 + V4 + V5",
        fixed_effects=["region"],
    )
    print(history)

.. code-block:: text

    ==============================================================================
     Dynamic Covariate Balancing History
    ==============================================================================

     Length         ATE          SE     mu(ds1)     mu(ds2)
     ------------------------------------------------------
          1      0.2743      0.2008      7.9699      7.6956
          2      0.3011      0.2032      8.0044      7.7033
          3      0.2385      0.2164      8.0130      7.7746
          4      0.2804      0.2205      8.0113      7.7309
          5      0.2451      0.2294      8.0095      7.7645

    ==============================================================================
     Viviano and Bradic (2026)

The effect ranges from 0.24 to 0.30 across history lengths, with the
estimated ``mu(ds1)`` stable around 8.0 and ``mu(ds2)`` fluctuating
between 7.70 and 7.77. Standard errors grow slightly with history length
as the
effective sample shrinks, since fewer countries maintain the same
democratic status for longer stretches. This trade-off between
identification of long-run effects and statistical precision is exactly
what the history length diagnostic is designed to reveal.

With this parsimonious specification (five covariates and region fixed
effects), the point estimates are positive and relatively stable across
horizons.
`Viviano and Bradic (2026) <https://doi.org/10.1093/biomet/asag016>`_
show that richer specifications using the full set of 158 covariates
with lagged outcomes produce larger and more precisely estimated effects,
with the ATE growing substantially at longer horizons. Their analysis
also demonstrates that standard local projections and the linear
regression approach of
`Acemoglu et al. (2019) <https://doi.org/10.1086/700936>`_ underestimate
the long-run effect because they average over the distribution of future
treatment assignments.

The result is a ``DynBalancingHistoryResult`` with a ``summary``
DataFrame and a list of individual ``DynBalancingResult`` objects, one
per history length. Each individual result exposes the same attributes
as the single-ATE result (``att``, ``se``, ``gammas``, ``coefficients``,
``imbalances``), allowing detailed diagnostics at each horizon.


Clustered standard errors
-------------------------

In applications where countries within the same geographic region may be
correlated, for example through trade linkages or shared institutions,
the ``clustervars`` argument produces cluster-robust standard errors
that are valid under arbitrary within-cluster dependence.

.. code-block:: python

    result_cl = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ V1 + V2 + V3 + V4 + V5",
        fixed_effects=["region"],
        clustervars=["region"],
    )
    print(f"Unclustered SE: {result.se:.4f}")
    print(f"Clustered SE:   {result_cl.se:.4f}")

.. code-block:: text

    Unclustered SE: 0.2032
    Clustered SE:   0.1479

Here clustering by region produces a smaller standard error, which can
happen when within-cluster residuals partially cancel, reducing the total
variance. The default is cross-sectionally independent standard errors,
matching the assumption in the theory. Which level to cluster at depends
on the empirical context and should reflect the structure of dependence
in the data.


Pooled regression
-----------------

By default, the estimator uses the outcome in the final period only. The
``pooled=True`` option pools observations across all periods into a
single regression with time fixed effects. This can improve precision
by increasing the effective sample size, at the cost of assuming that the
treatment effect is stationary across periods.

.. code-block:: python

    result_pooled = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ V1 + V2 + V3 + V4 + V5",
        fixed_effects=["region", "Time"],
        pooled=True,
    )
    print(f"Unpooled ATE: {result.att:.4f}  SE: {result.se:.4f}")
    print(f"Pooled ATE:   {result_pooled.att:.4f}  SE: {result_pooled.se:.4f}")

.. code-block:: text

    Unpooled ATE: 0.3011  SE: 0.2032
    Pooled ATE:   0.2895  SE: 0.2018

The pooled specification produces a similar point estimate (0.29 versus
0.30) with a slightly smaller standard error, consistent with the
increased effective sample size from pooling across periods.

When ``pooled=True`` and the time variable is included in
``fixed_effects``, the regression includes time dummies and standard
errors are automatically clustered at the unit level to account for
within-unit serial correlation, unless a larger clustering variable is
specified.


Balancing strategies
--------------------

While DCB is the recommended default, the ``balancing`` argument
supports several alternative weighting strategies: ``"ipw"`` for inverse
probability weighting, ``"aipw"`` for augmented IPW, and ``"ipw_msm"``
for marginal structural model weights.

.. code-block:: python

    result_ipw = dyn_balancing(
        data=df,
        yname="Y",
        tname="Time",
        idname="Unit",
        treatment_name="D",
        ds1=[1, 1],
        ds2=[0, 0],
        xformla="~ V1 + V2 + V3 + V4 + V5",
        fixed_effects=["region"],
        balancing="ipw",
    )

These alternatives require estimating the propensity score via logistic
regression, which can fail when covariates are high-dimensional or when
the propensity score is close to zero for some units. On the Acemoglu
et al. dataset, the AIPW estimator encounters a singular Hessian in the
propensity score model, illustrating exactly the instability that
motivates DCB. DCB avoids this problem entirely by constructing
balancing weights through a quadratic program that does not estimate or
require the propensity score.


Publication-quality tables
--------------------------

The result objects implement the ``maketables`` interface for combining
multiple specifications into a single publication table.

.. code-block:: python

    from moderndid.core.maketables import etable

    tab = etable(
        [result, result_cl, result_pooled],
        labels=["DCB", "DCB (clustered)", "DCB (pooled)"],
    )
    print(tab)

See :ref:`publication_tables` for more details on the table system.


Next steps
----------

For details on additional parameters including ``impulse_response``,
``final_periods``, ``debias``, and ``fast_adaptive``, see the
:ref:`API reference <api-diddynamic>`.

For theoretical background on the dynamic covariate balancing methodology
and the formal identification and inference results, see the
:ref:`Background <background-diddynamic>` section.
