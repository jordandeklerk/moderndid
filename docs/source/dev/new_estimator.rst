.. _new-estimator:

===================
Adding an estimator
===================

When you bring a method from a paper into moderndid, its estimating equation
tells you what to compute. It doesn't say which rows of the data make up one
observation, how sampling variation reaches the reported interval, or what the
result has to keep for the calculations that follow a fit. Mistakes in those
choices are easy to miss because an implementation can reproduce the paper's
point estimate while pairing the wrong outcomes or scaling its standard error
by the wrong count.

We'll work through the choices with a small estimator for a two-period panel,
the difference between the treated and comparison groups' mean outcome
changes. The package's outcome regression estimator, :func:`~moderndid.ordid`,
reduces to the same comparison without covariates and gives you an existing
estimator to check the new one against. By the end, the new estimator matches
it on a panel small enough to check by hand, covers a known effect at its
nominal rate in simulated data, and prints a report like the package's other
results.


Begin with the effect you want to estimate
------------------------------------------

Before deciding which arguments a function accepts, you need to be clear about
whose effect it estimates. Suppose each unit is observed once before and once
after a treatment that reaches the units with :math:`D_i=1` in the second
period. The comparison units with :math:`D_i=0` stay untreated in both periods.
The target is the average effect on the treated units in the second period,

.. math::

   \tau=\mathbb{E}[Y_{i1}(1)-Y_{i1}(0)\mid D_i=1].

The treated units' outcomes without treatment in that period are never
observed. The comparison group's mean outcome change stands in for the change
they would have seen. That substitution takes parallel trends in the untreated
outcomes, no anticipation of treatment, and no spillovers between the groups,
as the :doc:`two-period background <../background/drdid>` and Section 2.1 of
`Sant'Anna and Zhao (2020)
<https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_ explain. Assuming
parallel trends without conditioning on covariates and giving every unit the
same weight, we estimate the effect by the difference between the groups'
sample mean changes,

.. math::

   \widehat\tau
   =\overline{\Delta Y}_{D=1}-\overline{\Delta Y}_{D=0},
   \qquad \Delta Y_i=Y_{i1}-Y_{i0}.


.. _new-estimator-dispatch-table:
.. _new-estimator-register-builder:

Decide what one observation means
---------------------------------

Because the estimator compares each unit's two outcomes, one observation is a
unit with both of its outcomes. Long data hold those outcomes in two rows, one
for each period. Instead of pairing the rows ourselves, we'll use the
preparation that the package's two-period estimators share.
``preprocess_drdid`` matches each unit's rows by identifier and returns the
outcomes of the two periods as arrays ``y0`` and ``y1`` and the group indicator
``D``, all in one unit order.

.. code-block:: python

   import polars as pl

   from moderndid.core.preprocess import preprocess_drdid

   data = pl.DataFrame(
       {
           "unit": [1, 2, 3, 4, 4, 3, 2, 1],
           "year": [2000, 2000, 2000, 2000, 2001, 2001, 2001, 2001],
           "treated": [0, 0, 1, 1, 1, 1, 0, 0],
           "y": [10.0, 12.0, 20.0, 22.0, 28.0, 24.0, 15.0, 11.0],
       }
   )
   prepared = preprocess_drdid(data, yname="y", tname="year", treat_col="treated", idname="unit")

   print("Outcomes in 2000:", prepared.y0)
   print("Outcomes in 2001:", prepared.y1)
   print("Treated group:", prepared.D)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Outcomes in 2000: [10. 12. 20. 22.]
         Outcomes in 2001: [11. 15. 24. 28.]
         Treated group: [0 0 1 1]

Although the 2001 rows list the units in reverse order, the arrays line up by
unit. ``D`` records which group a unit belongs to rather than whether it was
treated in each period. A treated unit therefore carries a 1 even though its
first outcome came before treatment began.

.. admonition:: Row order doesn't pair units
   :class: warning

   Arrays cut from each year's rows in row order have equal lengths and can
   still hold different units in the same position. On these four units, that
   pairing turns the effect of 3 into one of -23. Since no check on shapes or
   values can catch the mistake, the pairing has to come from the identifiers.

Reusing ``preprocess_drdid`` also gives the new estimator the same sample as
the doubly robust estimator :func:`~moderndid.drdid`. On a panel, it drops a
unit observed in only one period with a warning. It stops with an error at a
missing outcome, a repeated unit and period, a treatment indicator that changes
within a unit, or a third period. For repeated cross-sections, the same
function returns one outcome per row and a post-period indicator instead.


Compute the effect with its influence function
----------------------------------------------

The routine that computes the effect should also return what inference needs.
In large samples, the estimate's error behaves like the average of one term per
unit, the unit's influence function. Those terms give the standard error and
later serve clustering, aggregation, and the bootstrap. With :math:`\widehat q`
the treated share and :math:`\widehat\mu_1` and :math:`\widehat\mu_0` the two
groups' mean outcome changes, unit :math:`i`'s estimated influence function is

.. math::

   \widehat\psi_i
   =\frac{D_i}{\widehat q}(\Delta Y_i-\widehat\mu_1)
    -\frac{1-D_i}{1-\widehat q}(\Delta Y_i-\widehat\mu_0).

Each unit contributes its deviation from its own group's mean change divided by
that group's share, as the :ref:`influence functions background
<background-influence-functions>` derives. Like the numerical routines
elsewhere in the package, the one below takes only arrays.

.. code-block:: python

   def compute_mean_did(y0, y1, d):
       """Compute the mean DiD and its influence function."""
       change = y1 - y0
       treated_mean = change[d == 1].mean()
       comparison_mean = change[d == 0].mean()
       treated_share = d.mean()

       treated_part = d / treated_share * (change - treated_mean)
       comparison_part = (1 - d) / (1 - treated_share) * (change - comparison_mean)
       return float(treated_mean - comparison_mean), treated_part - comparison_part

The routine returns the terms before any division by :math:`n`, under the
convention

.. math::

   \widehat\tau-\tau
   =\frac{1}{n}\sum_{i=1}^n\psi_i+o_p(n^{-1/2}).

The standard error therefore divides the square root of their summed squares
by the number of units,

.. math::

   \widehat{\operatorname{se}}(\widehat\tau)
   =\frac{1}{n}\Bigl(\sum_{i=1}^n\widehat\psi_i^2\Bigr)^{1/2}.

Like the analytical standard errors of the staggered adoption estimator
:func:`~moderndid.att_gt`, it applies no degrees-of-freedom correction.

.. admonition:: Count units, not rows
   :class: important

   The :math:`n` in the formula counts units, each with both of its outcomes,
   rather than rows of the long data. Dividing by the :math:`2n` rows would
   halve the standard error.

This standard error assumes that units are sampled independently of each
other. With clustered sampling, the :ref:`background on standard errors
<background-influence-functions-inference>` sums the terms within each cluster
before squaring. Fitted outcome or propensity score models add terms of their
own that the :ref:`background on fitted models
<background-influence-functions-fitted>` derives and the doubly robust panel
estimator :func:`~moderndid.drdid_panel` computes. Copy its formulas only where
your derivation gives the same terms.


.. _new-estimator-maketables:

Keep what the next calculation will need
----------------------------------------

Everything after the fit, from a printed report to an aggregation, reads the
result rather than the data. Besides the effect, the result therefore keeps the
interval's bounds, its critical value, the influence function, and the number
of units that scales it. We'll hold these in a ``NamedTuple`` like every other
moderndid result.

.. code-block:: python

   from typing import NamedTuple

   import numpy as np


   class MeanDIDResult(NamedTuple):
       """Result of the mean DiD estimator."""

       #: Average treatment effect on the treated.
       att: float
       #: Standard error of the effect.
       se: float
       #: Lower bound of the confidence interval.
       ci_lower: float
       #: Upper bound of the confidence interval.
       ci_upper: float
       #: Critical value of the interval.
       critical_value: float
       #: Influence function with one entry for each unit.
       influence_func: np.ndarray
       #: Number of units in the paired sample.
       n_units: int
       #: Significance level of the interval.
       alpha: float

In the package, the API reference shows the fields that the class docstring
lists under Attributes. The ``#:`` comments keep each field's meaning beside its
definition for readers of the code. The names match those of existing results,
such as ``influence_func`` and ``n_units`` on the result of
:func:`~moderndid.att_gt`.


Bring the choices into a callable estimator
-------------------------------------------

The entry point gathers the decisions above into the one function users call.
Its arguments reuse the :ref:`names other estimators give the same ideas
<consistent-argument-naming>`, from ``yname`` for the outcome column to
``treatname`` for the group indicator, as in :func:`~moderndid.ordid`. Before
any computation, it checks them with ``check_columns``, a shared helper that
names every column the data lack and suggests the closest match. The ``alpha``
check shows the wording every error message follows. It names the argument as
the user wrote it, says what is wrong, and says what to pass instead.

.. admonition:: Add an option once the calculation uses it
   :class: tip

   A covariate formula belongs in the signature only once the estimating
   equation adjusts for covariates. Adding one to this unadjusted comparison
   wouldn't make it valid under conditional parallel trends. Clustering, by
   contrast, would change only the standard error, the way ``alpha`` changes
   only the interval.

With a single unit, a group would contribute only a zero term and drop its
sampling variation from the standard error. We therefore require at least two
units in each group once preparation has settled the sample.

.. code-block:: python

   from scipy import stats

   from moderndid.core.preprocess.validators import check_columns


   def mean_did(data, yname, tname, idname, treatname, alpha=0.05):
       """Estimate the mean DiD of a two-period panel."""
       if not 0 < alpha < 1:
           raise ValueError(f"alpha={alpha} is not valid. Must be between 0 and 1 (exclusive).")
       check_columns(data, yname=yname, tname=tname, idname=idname, treatname=treatname)
       prepared = preprocess_drdid(data, yname=yname, tname=tname, treat_col=treatname, idname=idname)

       n_treated = int(prepared.D.sum())
       n_comparison = prepared.n_units - n_treated
       if min(n_treated, n_comparison) < 2:
           raise ValueError(
               f"treatname='{treatname}' gives {n_treated} treated and {n_comparison} "
               "comparison units. Pass data with at least two units in each group."
           )

       att, influence_func = compute_mean_did(prepared.y0, prepared.y1, prepared.D)
       se = float(np.sqrt(np.sum(influence_func**2)) / prepared.n_units)
       critical_value = float(stats.norm.ppf(1 - alpha / 2))
       return MeanDIDResult(
           att=att,
           se=se,
           ci_lower=att - critical_value * se,
           ci_upper=att + critical_value * se,
           critical_value=critical_value,
           influence_func=influence_func,
           n_units=prepared.n_units,
           alpha=alpha,
       )

On the four units from earlier, the new estimator and :func:`~moderndid.ordid`
should agree on the effect, its standard error, and every unit's term.

.. code-block:: python

   import moderndid as did

   columns = {"yname": "y", "tname": "year", "idname": "unit", "treatname": "treated"}
   result = mean_did(data, **columns)
   reference = did.ordid(data, **columns, inf_func=True)

   print("Influence function:", result.influence_func)
   print(f"ATT: {result.att:.4f}, outcome regression {reference.att:.4f}")
   print(f"Standard error: {result.se:.4f}, outcome regression {reference.se:.4f}")
   print("Same influence function:", np.allclose(result.influence_func, reference.att_inf_func))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Influence function: [ 2. -2. -2.  2.]
         ATT: 3.0000, outcome regression 3.0000
         Standard error: 1.0000, outcome regression 1.0000
         Same influence function: True

The squares of the four printed terms add up to 16. Dividing its square root by
the 4 units gives the standard error of 1 that both estimators report.
Agreement on four units checks the arithmetic and the scaling but can't say
whether a 95 percent interval covers the effect in 95 percent of samples.


Recover a known effect in simulated data
----------------------------------------

Checking coverage takes many samples with a known effect, because coverage
describes the estimator across samples rather than in one fit. In each
simulated panel below, about 40 percent of the 1,000 units are treated. They
start from a higher outcome level than the comparison units but share their
untreated trend of 1. Treatment adds 2 to the second outcome of every treated
unit. We'll fit 500 such panels and compare the spread of the estimates with
the standard errors the fits report.

.. code-block:: python

   def simulate_panel(n_units, effect, rng):
       """Draw a two-period panel in which treatment raises the second outcome by effect."""
       treated = rng.binomial(1, 0.4, n_units)
       level = rng.normal(2 * treated, 1)
       y0 = level + rng.normal(size=n_units)
       y1 = level + 1 + effect * treated + rng.normal(size=n_units)
       return pl.DataFrame(
           {
               "unit": np.tile(np.arange(n_units), 2),
               "year": np.repeat([2000, 2001], n_units),
               "treated": np.tile(treated, 2),
               "y": np.concatenate([y0, y1]),
           }
       )


   rng = np.random.default_rng(0)
   fits = [mean_did(simulate_panel(1000, 2.0, rng), **columns) for _ in range(500)]
   estimates = np.array([fit.att for fit in fits])
   covered = [fit.ci_lower <= 2.0 <= fit.ci_upper for fit in fits]

   print(f"Mean estimate: {estimates.mean():.3f}")
   print(f"Standard deviation of the estimates: {estimates.std():.3f}")
   print(f"Mean standard error: {np.mean([fit.se for fit in fits]):.3f}")
   print(f"Coverage of the 95 percent intervals: {np.mean(covered):.3f}")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Mean estimate: 2.004
         Standard deviation of the estimates: 0.093
         Mean standard error: 0.091
         Coverage of the 95 percent intervals: 0.950

Across the 500 panels, the estimates center on the true effect of 2. Their
standard deviation of 0.093 is close to the average standard error of 0.091
that the fits report. The intervals cover the true effect in 95 percent of the
panels. With 500 panels, any coverage between about 93 and 97 percent would be
consistent with a correct interval.

For a method with nuisance models, :func:`~moderndid.gen_did_scalable` plants
known effects in staggered data and lets you choose which of the outcome and
propensity score models are correctly specified.


Print a report users can read
-----------------------------

Users read a result through its printed report rather than its raw fields. A
formatter builds that report from the helpers in ``moderndid.core.format``.
Those helpers give the new report the same title bars, effect table,
significance note, and footer as the package's other reports. ``attach_format``
then makes the formatter supply the text whenever the result is printed or
displayed.

.. code-block:: python

   from moderndid.core.format import (
       attach_format,
       format_footer,
       format_significance_note,
       format_single_result_table,
       format_standard_errors,
       format_title,
   )


   def format_mean_did_result(result):
       """Format a mean DiD result for display."""
       conf_level = round((1 - result.alpha) * 100)
       lines = format_title("Mean Difference-in-Differences")
       lines.append(f" Computed from {result.n_units} units.")
       lines.extend(
           format_single_result_table(
               "ATT", result.att, result.se, conf_level, result.ci_lower, result.ci_upper
           )
       )
       lines.extend(format_significance_note())
       lines.append(format_standard_errors(bootstrap=False))
       lines.extend(format_footer("Reference: Sant'Anna and Zhao (2020)"))
       return "\n".join(lines)


   attach_format(MeanDIDResult, format_mean_did_result)
   print(result)

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         ==============================================================================
          Mean Difference-in-Differences
         ==============================================================================
          Computed from 4 units.

         ┌────────┬────────────┬────────────────────────┐
         │    ATT │ Std. Error │ [95% Conf. Interval]   │
         ├────────┼────────────┼────────────────────────┤
         │ 3.0000 │     1.0000 │ [  1.0400,   4.9600] * │
         └────────┴────────────┴────────────────────────┘

         ------------------------------------------------------------------------------
          Signif. codes: '*' confidence interval does not cover 0
          Analytical standard errors
         ==============================================================================
          Reference: Sant'Anna and Zhao (2020)

Because the report reads the interval and significance level stored in the
result instead of recomputing them, it always shows the interval the fit
formed. Its footer names the paper behind the method, the one place outside the
entry point's References section where authors appear.

.. admonition:: Register the formatter on import
   :class: important

   The ``attach_format`` call at the bottom of ``format.py`` runs only when that
   module is imported. A family's ``__init__.py`` therefore imports
   ``format.py`` with a relative import. Without that import, every result
   prints its raw fields instead of the report.


.. _new-estimator-module-layout:

Bring the estimator into the package
------------------------------------

Each function on this page has a home in an estimator family, the folder under
``moderndid/`` that holds one method. In the ``did`` family, ``att_gt.py``
holds the entry point, ``compute_att_gt.py`` the numerical routines,
``container.py`` the results, and ``format.py`` the reports. A public entry
point also needs a References section that cites the paper, exports in the
family's and the root ``__init__.py``, an API page, a converter that lets
:func:`~moderndid.to_df` turn its result into a table, and a ``README.md`` for
the family. The :doc:`architecture guide <architecture>` shows how the exports
and the converter connect to the rest of the package.

.. admonition:: Copy the did layout, not drdid's
   :class: note

   Some older families, such as ``drdid``, define a result type beside each
   estimator. A new family keeps every result type in its ``container.py``.

A staggered adoption estimator repeats this comparison for each cohort and
period. Its result labels every effect and keeps one influence function column
per effect over the same units. When an aggregation of those effects estimates
its weights, the :ref:`staggered DiD background
<background-did-aggregation-weights>` derives the term those weights add.

The :doc:`testing guide <../contributing/testing>` shows how to keep checks
like these in the test suite, from a hand-computed case to a seeded
simulation. If a number later disagrees with what you expected, the
:doc:`debugging guide <debugging>` traces the difference to the sample, the
effect, or its uncertainty.
