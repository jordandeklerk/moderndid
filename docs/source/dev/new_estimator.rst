.. _new-estimator:

===================
Adding an estimator
===================

When you bring a method from a paper into moderndid, the estimating equation
is a starting point for the implementation. You still need to decide what
one observation means, how the data will reach that equation, and what
uncertainty the result should report. We'll work through those choices by
developing a two-period panel estimator, from the effect we want to estimate
to a function we can call on paired outcomes.

The example uses the familiar difference between two groups' mean outcome
changes. You can already estimate this effect with
:func:`~moderndid.reg_did_panel` without covariates, so we can follow the
calculation without introducing nuisance models at the same time. The code
below builds a small working estimator; the accompanying discussion explains
which choices you would revisit for another method. :doc:`architecture`
covers how those pieces connect to the rest of the package once the
statistical implementation is ready.


Begin with the effect you want to estimate
------------------------------------------

Before deciding which arguments a function accepts, we need to be clear
about whose effect it will estimate. Suppose we observe the same units
before and after a treatment begins. Units with :math:`D_i=1` receive
treatment in the second period and units with :math:`D_i=0` remain
untreated in both periods. Our target is the average effect on the treated
units after treatment begins,

.. math::

   \tau=\mathbb{E}[Y_{i1}(1)-Y_{i1}(0)\mid D_i=1].

Since we cannot observe the treated units' outcomes without treatment in
that second period, we use the comparison group's mean outcome change to
recover their counterfactual change. This requires parallel mean untreated
trends across the groups, no anticipation, and no spillovers. You can follow
the identification argument in the
:doc:`two-period background <../background/drdid>` and in Section 2 of
`Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.

For this implementation, we assume parallel trends without conditioning on
covariates and give every unit equal weight. Those choices lead to the
sample difference in mean changes,

.. math::

   \widehat\tau
   =\overline{\Delta Y}_{D=1}-\overline{\Delta Y}_{D=0},
   \qquad \Delta Y_i=Y_{i1}-Y_{i0}.

If your method instead uses conditional parallel trends, the adjustment
for covariates belongs in this calculation. Adding a formula argument to
the function would not make the unadjusted comparison above appropriate
for that design. Starting from the target and identification assumptions
helps you decide what needs to enter the computation before choosing an
existing implementation to build on.


.. _new-estimator-dispatch-table:
.. _new-estimator-register-builder:

Decide what one observation means
---------------------------------

Because this is a panel estimator, one observation in the calculation is a
paired unit rather than an outcome row. We'll give the numerical routine
three arrays, ``y0``, ``y1``, and ``d``, whose positions refer to the same
units. The first two contain outcomes before and after treatment; ``d``
records treatment-group membership rather than treatment status in each
period. A treated unit therefore has a 1 in ``d`` even though its first
outcome was observed before treatment began.

The distinction matters when you prepare the arrays from a DataFrame.
Extracting the periods without matching unit identifiers can leave arrays
with equal lengths but different units in the same position. In moderndid, the existing
``TwoPeriodDIDConfig`` and ``PreprocessDataBuilder`` prepare paired outcomes
as ``y0``, ``y1``, and ``D``. The
:doc:`data preparation discussion <architecture>` explains that path and
how to choose another layout when a method requires it. Repeated
cross-sections, for example, cannot supply the within-unit changes used here.

We also need a deliberate policy for an incomplete sample. This example
rejects missing or non-finite values and requires at least two units in each
group. If an implementation instead drops incomplete units, that decision
can change the population represented by the fit and needs to be explained to
the caller. At the array boundary, the checks can establish compatible
shapes and valid values; correct pairing still has to come from preparation.


Compute the effect and its uncertainty together
-----------------------------------------------

Once the paired sample is settled, the point estimate follows directly
from the two mean changes. We also need to work out how sampling variation
in those means reaches the estimated effect, rather than add a standard
error after the calculation is finished. For the example, we'll use
analytical inference under independent and identically distributed unit
sampling, finite outcome-change variance, and positive population shares
for both groups.

Let :math:`\widehat p` be the treated share and let
:math:`\widehat\mu_1` and :math:`\widehat\mu_0` be the groups' mean outcome
changes. The estimated influence contribution for unit :math:`i` is

.. math::

   \widehat\psi_i
   =\frac{D_i}{\widehat p}(\Delta Y_i-\widehat\mu_1)
    -\frac{1-D_i}{1-\widehat p}(\Delta Y_i-\widehat\mu_0).

For each unit, the contribution uses its outcome change relative to its
own group's mean and divides that deviation by the group's sample share.
This expresses the sampling variation of both group means relative to the
full number of units. We can return those contributions alongside the
contrast from a numerical routine that does not need to know how the
original columns were named.

.. code-block:: python

   import numpy as np


   def compute_mean_did(y0, y1, d):
       """Compute the contrast and its unit influence contributions."""
       change = y1 - y0
       treated_mean = change[d == 1].mean()
       comparison_mean = change[d == 0].mean()
       treated_share = d.mean()

       treated_component = d / treated_share * (change - treated_mean)
       comparison_component = (1 - d) / (1 - treated_share) * (change - comparison_mean)
       influence = treated_component - comparison_component
       att = float(treated_mean - comparison_mean)
       return att, influence

To use those contributions for inference, the calling function needs to
know how they are scaled. We'll follow the convention

.. math::

   \widehat\tau-\tau
   =\frac{1}{n}\sum_{i=1}^n\psi_i+o_p(n^{-1/2}),

we retain the contributions before division by :math:`n` and compute

.. math::

   \widehat{\operatorname{se}}(\widehat\tau)
   =\frac{\sqrt{\sum_{i=1}^n\widehat\psi_i^2}}{n}.

Since each contribution describes a paired unit, :math:`n` counts those
units rather than the outcome rows in the original panel. Dividing by the
row count would therefore change the standard error's normalization.
The formula uses the influence-function variance convention without a
degrees-of-freedom correction; if your method uses another convention,
its calculation needs to reflect that choice.

That sampling decision has consequences for a more complex design. If
counties share shocks within a state, treating every county as independent
would leave out dependence that inference needs to preserve. Similarly,
when a method fits outcome or propensity models, their estimation may
contribute to the influence function. The panel implementations of
:func:`~moderndid.reg_did_panel` and :func:`~moderndid.drdid_panel` show how
those contributions enter covariate-adjusted estimators. Their formulas
need to follow the new method's derivation before you reuse them.


.. _new-estimator-maketables:

Keep what the next calculation will need
----------------------------------------

With the effect and its influence contributions available, we can decide
what the fitted result should retain. A caller who reads the effect now
may need its uncertainty for a later report or calculation. We'll keep the standard
error, fitted interval and critical value, unit contributions, sample count,
and significance level together in a small ``NamedTuple``.

.. code-block:: python

   from typing import NamedTuple


   class MeanDIDResult(NamedTuple):
       """A panel effect with its independent-unit analytical inference."""

       att: float
       se: float
       ci_lower: float
       ci_upper: float
       critical_value: float
       influence_func: np.ndarray
       n_units: int
       alp: float

A later report can display the interval used in the fit because the result
retains its bounds and critical value. For a DataFrame-facing estimator,
the unit contributions also need their corresponding identifiers so
filtering or reordering cannot silently change what they refer to.
Once these quantities are settled, :doc:`architecture` explains how
containers make them available to printed reports, DataFrames, and
publication tables.


Bring the choices into a callable estimator
-------------------------------------------

We can now assemble the entry point from the pieces above. It accepts the
paired arrays and a significance level, checks the sample policy we chose,
and passes valid arrays to ``compute_mean_did``. The returned contributions
supply the analytical standard error; a normal critical value at the
requested level completes the result.

.. code-block:: python

   from scipy import stats


   def mean_did(y0, y1, d, alp=0.05):
       """Estimate an unweighted DiD from aligned paired outcomes."""
       y0, y1, d = (np.asarray(values, dtype=float) for values in (y0, y1, d))
       if any(values.ndim != 1 for values in (y0, y1, d)):
           raise ValueError("Outcomes and group membership must be one-dimensional.")
       if y0.shape != y1.shape or y0.shape != d.shape:
           raise ValueError("Outcomes and group membership must have matching shapes.")
       if not all(np.isfinite(values).all() for values in (y0, y1, d)):
           raise ValueError("The sample must contain only finite values.")
       if not np.isin(d, [0, 1]).all():
           raise ValueError("Group membership must contain only 0 and 1.")
       if (d == 1).sum() < 2 or (d == 0).sum() < 2:
           raise ValueError("Each group must contain at least two units.")
       if not 0 < alp < 1:
           raise ValueError("alp must be between zero and one.")

       att, influence = compute_mean_did(y0, y1, d)
       n_units = len(d)
       se = float(np.linalg.norm(influence) / n_units)
       critical_value = float(stats.norm.ppf(1 - alp / 2))
       return MeanDIDResult(
           att=att,
           se=se,
           ci_lower=att - critical_value * se,
           ci_upper=att + critical_value * se,
           critical_value=critical_value,
           influence_func=influence,
           n_units=n_units,
           alp=alp,
       )

The function exposes only choices that the calculation implements. If you
later support sampling weights or clustering, those arguments need to
change the estimating equation or inference as well as the information
stored in the result. The :ref:`argument conventions
<consistent-argument-naming>` help you keep their names familiar to callers
without borrowing options that the method does not yet support.


Follow one calculation through the result
------------------------------------------

A small panel lets us see whether the function carries the intended
comparison through to its result. We'll give the comparison units outcome
changes of 1 and 3, and the treated units changes of 4 and 6. Their means
are 2 and 5; subtracting them gives an expected effect of 3 outcome units.

.. code-block:: python

   y0 = np.array([10, 12, 20, 22])
   y1 = np.array([11, 15, 24, 28])
   d = np.array([0, 0, 1, 1])

   result = mean_did(y0, y1, d)
   print(f"ATT: {result.att:.4f}")
   print(f"Standard error: {result.se:.4f}")
   print(f"95 percent interval: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
   print(f"Paired units: {result.n_units}")
   print(f"Influence contributions: {result.influence_func}")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         ATT: 3.0000
         Standard error: 1.0000
         95 percent interval: [1.0400, 4.9600]
         Paired units: 4
         Influence contributions: [ 2. -2. -2.  2.]

Alongside the expected effect of 3 outcome units, the result retains four
influence contributions that sum to zero. Their squared sum of 16 gives a
standard error of :math:`\sqrt{16}/4=1`. The interval uses the fitted normal critical
value of about 1.96. This constructed panel checks the calculation; four
units do not establish the empirical coverage of a large-sample interval.

Changing ``alp`` would change the interval without changing the point
estimate or unit contributions. Choosing a different comparison or fitting
covariate models would instead change the statistical calculation itself.
That distinction helps you decide whether a new option belongs in the
numerical routine, the inference procedure, or the preparation of the sample.


.. _new-estimator-module-layout:

Develop the method beyond this example
--------------------------------------

You can choose an existing implementation to build on by looking for the
data layout and identifying comparison closest to your method.
``reg_did_panel.py`` follows paired changes through an
outcome regression. Staggered adoption methods organize many such
comparisons by cohort and time; a continuous-treatment method also needs
to describe the dose at which an effect is evaluated.

For those methods, the result needs labels for its effect axes so a caller
can tell which comparison each estimate represents. If an aggregation estimates its weights, their
uncertainty needs to reach the aggregate alongside the uncertainty in its
component effects. The result therefore needs the information required
by that aggregation's derivation as well as the component estimates.

When the method is ready to live in the package, keep its numerical routine
separate from the public entry point and place its result type in the
family's ``container.py``. That separation lets you follow a change in the
estimating equation without also tracing DataFrame preparation or reporting.
:doc:`architecture` provides the implementation map for those interfaces;
:doc:`../contributing/testing` covers the checks that support a contribution.
