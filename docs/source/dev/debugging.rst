.. _debugging:

Debugging an analysis
=====================

When a fit comes back with warnings, with effects missing, or with numbers that
disagree with a reference, the cause sits in one of three layers. It can be in
the units that enter each comparison, in the estimator that a cell runs on
them, or in the uncertainty around the result. This guide shows how to find the
layer before you change the code, the data, or an expected value, since a
change made before you know the cause can hide a real error.

We'll follow one investigation from the first report to its cause. When a
colleague adds a dummy for the counties of Illinois to control for that state's
labor market, every group-time effect in the minimum wage analysis disappears.
By the end, you'll have checked what each comparison held, rebuilt one
comparison on its own, and found why no estimator can fit it.

Reproduce the failure in a script
---------------------------------

The investigation starts from the smallest script that shows the problem. In
the county data, a cohort holds the counties whose state first raised its
minimum wage in the same year. :func:`~moderndid.att_gt` estimates one effect
for each cohort in each year from 2004 to 2007. Each of these 12 pairs of a
cohort and a year is a cell. We turn off the bootstrap and the uniform
confidence bands so that no random draws play a part. The script also records
the warnings so that it can count them.

.. code-block:: python

   import warnings

   import moderndid as did
   import numpy as np
   import polars as pl

   # The colleague's dummy marks the counties of Illinois, whose state code is 17.
   data = did.load_mpdta().with_columns(
       illinois=(pl.col("countyreal") // 1000 == 17).cast(pl.Int64)
   )
   spec = {
       "yname": "lemp",
       "tname": "year",
       "idname": "countyreal",
       "gname": "first.treat",
       "boot": False,
       "cband": False,
   }

   # Record the warnings so that the script can count them.
   with warnings.catch_warnings(record=True) as caught:
       warnings.simplefilter("always")
       failing = did.att_gt(data, **spec, xformla="~ lpop + illinois")

   messages = [str(w.message) for w in caught]
   print("Effects estimated:", len(failing.att_gt))
   print("Cells short of control units:", sum("Not enough control units" in m for m in messages))
   print("Cells that violate overlap:", sum("Overlap condition violated" in m for m in messages))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Effects estimated: 0
         Cells short of control units: 12
         Cells that violate overlap: 4

Each of the 12 cells failed with the same warning about its own cohort and
year. For the first cell it reads "Not enough control units for group 2004 in
time period 2004 to run specified regression". The four cells of the 2004
cohort also report that overlap is violated. :func:`~moderndid.aggte` refuses
the empty result with an error that points back to these warnings. If a
failing test reported the problem instead, the :doc:`testing guide
<../contributing/testing>` shows how to run it alone. Adding ``-x --pdb`` to
that command stops in Python's debugger at the first failure.

.. admonition:: Show the warnings a test run hides
   :class: tip

   ``tests/conftest.py`` tells pytest to ignore a few numerical and dependency
   warnings, such as statsmodels' perfect separation warning. Adding
   ``-W default`` to the pytest command shows them again. To find the line that
   raises a warning, turn warnings into errors with
   ``warnings.simplefilter("error")`` and read the traceback.

Check what each cell compared
-----------------------------

Since a shortage of control units would mean that the comparisons ran out of
counties, the next check counts them. With no cells left in the failing fit,
the counts come from the fit without the dummy. Its influence functions hold
one row per county and one column per cell. Because a county outside a cell's
cohort and comparison group contributes exactly zero to that column, the
nonzero entries count the counties that the cell compared.

.. code-block:: python

   # The same analysis without the dummy.
   original = did.att_gt(data, **spec, xformla="~ lpop")

   # Count the counties with a nonzero contribution to each cell.
   compared = did.to_df(original).with_columns(
       counties=np.count_nonzero(original.influence_func, axis=0)
   )
   print(compared.pivot("time", index="group", values="counties"))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (3, 5)
         ┌───────┬──────┬──────┬──────┬──────┐
         │ group ┆ 2004 ┆ 2005 ┆ 2006 ┆ 2007 │
         │ ---   ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
         │ i64   ┆ i64  ┆ i64  ┆ i64  ┆ i64  │
         ╞═══════╪══════╪══════╪══════╪══════╡
         │ 2004  ┆ 329  ┆ 329  ┆ 329  ┆ 329  │
         │ 2006  ┆ 349  ┆ 349  ┆ 349  ┆ 349  │
         │ 2007  ┆ 440  ┆ 440  ┆ 440  ┆ 440  │
         └───────┴──────┴──────┴──────┴──────┘

Each cell compares the 309 never-treated counties with the 20 to 131 counties
of its cohort. With that many counties in every comparison, the shortage that
the warnings describe can't be a matter of numbers. The same counts tell you,
when two fits disagree, whether they compared the same counties at all.

Rebuild one cell on its own
---------------------------

Cell (2006, 2006), like every cell, is a two-period comparison. It sets the
change in teen employment from 2005 to 2006 in the 2006 cohort against the
change in the never-treated counties. Running that comparison on those rows
alone with :func:`~moderndid.drdid` turns a 12-cell fit into one small problem
that you can inspect line by line. Before you trust the rebuilt cell with the
failing specification, check that it reproduces the working one.

.. code-block:: python

   # Cell (2006, 2006) compares the 2006 cohort with the never-treated counties from 2005 to 2006.
   rows = data.filter(pl.col("first.treat").is_in([0, 2006]) & pl.col("year").is_in([2005, 2006]))
   rows = rows.with_columns(treated=(pl.col("first.treat") == 2006).cast(pl.Int64))
   cell = {
       "yname": "lemp",
       "tname": "year",
       "idname": "countyreal",
       "treatname": "treated",
       "est_method": "trad",
   }

   rebuilt = did.drdid(rows, **cell, xformla="~ lpop")
   index = np.flatnonzero((original.groups == 2006) & (original.times == 2006))[0]
   print(f"Rebuilt: {rebuilt.att:.6f} (standard error {rebuilt.se:.6f})")
   print(f"att_gt:  {original.att_gt[index]:.6f} (standard error {original.se_gt[index]:.6f})")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         Rebuilt: 0.000961 (standard error 0.019400)
         att_gt:  0.000961 (standard error 0.019400)

The rebuilt cell matches the estimate and standard error of ``att_gt`` to every
printed digit. The same rebuild matches all 12 cells up to rounding.

.. admonition:: Rebuild a cell with the estimator it used
   :class: important

   Since ``att_gt`` estimates its doubly robust cells with the traditional form
   of the estimator, a rebuild with ``drdid`` passes ``est_method="trad"``. Left
   at its default of ``"imp"``, ``drdid`` runs the improved form instead. On the
   county data, the improved form moves a cell's estimate by as much as 0.00014
   and could send you looking for a bug that isn't there.

Once the dummy enters its formula, the rebuilt cell fails as well and gives a
message that names the cause.

.. code-block:: python

   did.drdid(rows, **cell, xformla="~ lpop + illinois")

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         UserWarning: The propensity score model dropped 1 covariate that is collinear with the others.
         ValueError: The regression design matrix of the comparison group is singular. Consider removing covariates that are constant or collinear among the comparison units.

The propensity score model drops the dummy because it adds nothing to the
other covariates. The outcome regression that ``drdid`` fits on the comparison
counties keeps every covariate and stops on its singular design.

Find the cause in the cell's data
---------------------------------

A singular design means that some covariate repeats what the others already
say among the comparison counties. Counting the cell's counties by the dummy
shows which one.

.. code-block:: python

   # How the dummy splits the counties of this cell in its first year.
   print(rows.filter(pl.col("year") == 2005).group_by("treated", "illinois").len().sort("treated"))

.. container:: cell_output

   .. container:: output stream

      .. code-block:: text

         shape: (2, 3)
         ┌─────────┬──────────┬─────┐
         │ treated ┆ illinois ┆ len │
         │ ---     ┆ ---      ┆ --- │
         │ i64     ┆ i64      ┆ u32 │
         ╞═════════╪══════════╪═════╡
         │ 0       ┆ 0        ┆ 309 │
         │ 1       ┆ 0        ┆ 40  │
         └─────────┴──────────┴─────┘

Every county in this cell, treated or not, has a dummy of 0. A column that
never changes adds nothing to the intercept. The comparison regression is then
left with one coefficient more than it can identify. The cells of the 2004
cohort have a second problem. Since all 20 of its counties are in Illinois, the
dummy matches treatment exactly in those cells. That perfect prediction is the
overlap violation that the fit reported for them.

Decide what to change
---------------------

Both failures trace back to the specification rather than to the code. A
covariate asks the estimator to compare counties that share its value. With
the dummy in the formula, the counties of Illinois need never-treated counties
in Illinois as their comparison. Since every never-treated county lies outside
Illinois, no such comparison exists. In every other cell the dummy is 0
throughout and gives the estimator nothing to compare.

Dropping the dummy fixes both failures and costs the analysis nothing that the
data could support. Any difference in the labor market of Illinois that stays
fixed over time already cancels in the change in employment that each cell
compares.

.. admonition:: Find the cause before a numerical fix
   :class: important

   Clipping propensity scores, adding regularization, or changing which units a
   fit keeps can make a failing calculation run. Each of these fixes also
   changes what the estimator measures. Establish why the calculation fails
   before you reach for one.

Rebuilding the cell paid off because the error from ``drdid`` names the cause
behind the warning from ``att_gt``. When an investigation does end in the code,
the fix comes with a test that fails without it. The :ref:`testing guide
<testing-numerical-tolerances>` describes how to choose that test's expected
value and tolerances.

Effects before uncertainty
--------------------------

When two fits differ, settle the point estimates first, the influence functions
behind them next, and the standard errors built from those last. Two fits that
agree on every effect and every influence function but not on the standard
errors point to the variance step. A clustered fit, for example, sums the
contributions within each cluster before it forms the variance. If two fits
differ only in their uniform confidence bands, the cause is the critical value
that the bands take from bootstrap draws. Since ``aggte`` makes those draws
even when ``boot=False``, two runs agree only when each gets the same integer
``random_state``. The :doc:`architecture guide <architecture>` rebuilds the
standard errors from the influence functions.

The same checks apply when one result comes from a GPU or from Spark. The
:ref:`GPU section of the testing guide <testing-gpu>` and :doc:`fits on Spark
<spark_fits>` show how far those results agree with the in-memory fit and why.
