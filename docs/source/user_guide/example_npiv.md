---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(example_npiv)=

# Nonparametric instrumental variables

This example steps outside difference-in-differences to trace how the share of a
household's budget spent on food changes as its total spending rises. In the
nineteenth century the statistician Ernst Engel found that poorer families spend
a larger share of their budgets on food. Among the 1,655 British couples we'll
study, the question is whether the data can rule out a food share that stays the
same as total spending rises and, if they can, how fast the share falls.

The simplest way to draw this curve, a line or a quadratic in log spending,
fixes its shape before the data have a say. Any regression of the food share on
total spending, however flexible, carries a subtler flaw. It treats spending as
if it arrived from outside the household, when the same tastes that decide how a
household splits its budget can also shape how much it spends in total.

{func}`~moderndid.npiv` addresses the first flaw with splines whose number of
segments the data choose and the second with the earnings of each household's
head as an instrument for its total spending. It's the same estimator that lets
the data pick the spline in the last check of the
{ref}`continuous treatment example <example_cont_did>`. You'll come away with
the curve and a band that covers all of it at once, the slope and food
elasticity the curve implies, and a set of checks that each change one choice.

```{code-cell} ipython3
:tags: [remove-cell]

from plotnine import options

options.figure_size = (12, 5)
options.dpi = 100
```

## Household budgets in 1995

The 1995 British Family Expenditure Survey recorded what each of these couples
spent on every group of goods, along with their incomes and family makeup. To keep
the households alike,
[Blundell, Chen, and Kristensen (2007)](https://doi.org/10.1111/j.1468-0262.2007.00808.x)
chose married or cohabiting couples with at most two children. They also dropped
households whose head was out of work, since the head's earnings serve as the
instrument for total spending. Their case for an instrument is what lets these
data answer the question. In their argument, earnings move how much a household
can spend but have nothing to do with what it prefers to buy.

```{code-cell} ipython3
import moderndid as did
import numpy as np

data = did.load_engel()
data.select("food", "logexp", "logwages").head()
```

Each of the 1,655 rows that {func}`~moderndid.load_engel` returns describes one
household by its budget shares for seven groups of goods, its total spending, its
earnings, and whether it has children. Three of those columns matter here, `food`
for the share of spending on food eaten at home, `logexp` for the log of total
spending on nondurable goods and services, and `logwages` for the instrument.
Spending on meals out goes into a separate `catering` column instead.

:::{admonition} What the logwages column holds
:class: note

Despite its name, `logwages` records the log of the household head's gross earnings
before taxes rather than an hourly wage.
:::

Since a curve can only be pinned down where there are households, the next cell
checks how they spread along total spending and how closely earnings follow it.

```{code-cell} ipython3
# How the households spread over log total spending and how closely earnings follow it.
spending = data["logexp"].to_numpy()
earnings = data["logwages"].to_numpy()
low, high = np.percentile(spending, [1, 99])
under = np.sum(spending < 4.5)
over = np.sum(spending > 6.5)
correlation = np.corrcoef(spending, earnings)[0, 1]

print(f"{len(spending)} households, log spending from {spending.min():.2f} to {spending.max():.2f}")
print(f"1st and 99th percentiles {low:.2f} and {high:.2f}")
print(f"{len(spending) - under - over} between 4.5 and 6.5, {under} below and {over} above")
print(f"correlation of log spending and log earnings {correlation:.3f}")
```

Although log total spending runs from 3.61 to 7.43, only 24 households fall below
4.5 and only 24 above 6.5. The other 1,607 lie between 4.5 and 6.5, two values
close to the 1st and 99th percentiles of 4.45 and 6.57. With a correlation of
0.514 between the two logs, households that earn more tend to spend more as well.
That link is the first thing an instrument for total spending has to show.

## The structural Engel curve

The curve we're after, $h_0$, links the food share $Y$ to log total spending $X$.
With log earnings $W$ as the instrument, it's pinned down by requiring the part of
the food share it leaves unexplained to average zero at every level of earnings,

$$
\mathbb{E}\big[\,Y - h_0(X) \mid W\,\big] = 0.
$$

Read that way, $h_0(x)$ is the average food share households would have if their
total spending were set to $x$. A regression of $Y$ on $X$ answers a different
question, the average food share among households that happen to spend $x$. The
two differ whenever $Y - h_0(X)$ moves with $X$, as it does when the decisions that
set total spending also shape the food share.

Engel's law is a statement about which way $h_0$ slopes as total spending rises.
Because food spending is the food share times total spending, the elasticity of
food spending with respect to total spending is $1 + h_0'(x)/h_0(x)$. Since the
share is positive, the elasticity falls below 1, the mark of a necessity, exactly
where the curve slopes down.

Recovering $h_0$ from these households rests on four assumptions about the
instrument, total spending, and the curve.

- Earnings are unrelated to the tastes behind the food share and change it only
  through total spending.
- Earnings shift total spending strongly enough to tell any two candidate curves
  apart.
- The curve is smooth enough for cubic splines to approximate both it and its
  slope.
- Whatever else drives the food share adds to the curve instead of changing its
  shape.

:::{admonition} How strong the instrument must be
:class: note

A linear instrument only has to shift average total spending up or down. Here
earnings must also tell apart curves that differ in fine detail, a job that gets
harder as the detail gets finer.
:::

The [background page on nonparametric instrumental variables](../background/npiv)
states the model formally and defines the measure of instrument strength behind the
second assumption. To estimate the curve, {func}`~moderndid.npiv` approximates
$h_0$ with B-splines and fits their coefficients by two-stage least squares on
B-splines of earnings. Two procedures from
[Chen, Christensen, and Kankanala (2024)](https://doi.org/10.1093/restud/rdae025)
decide how many spline segments to use and how to build bands around the result.

## Settling the specification

The estimation turns on the instrument, on how flexible the curve may be, and on
where to read it. The dictionary below gives each of the three its own group of
arguments. [How much each choice matters](#how-much-each-choice-matters) later
varies each of those arguments while the rest keep the values set below.

```{code-cell} ipython3
# Evaluate the curve at 81 points from 4.5 to 6.5, one every 0.025 of log total spending.
grid = np.linspace(4.5, 6.5, 81)

# The specification, grouped by the question each argument answers.
spec = dict(
    # Instrument log total spending with log earnings.
    yname="food",
    xname="logexp",
    wname="logwages",
    # Cubic splines whose number of segments the data choose, and quartic splines on four times
    # as many segments for the instrument.
    j_x_degree=3,
    j_x_segments=None,
    k_w_degree=4,
    k_w_smooth=2,
    # Compare the candidates and draw 95 percent bands over the middle of the spending range.
    x_grid=grid,
    x_eval=grid,
    alpha=0.05,
    biters=1000,
    seed=7,
)
```

### Earnings as the instrument for total spending

Since total spending is itself a household decision, the estimation needs
something that moves it without moving the food share directly. With
`wname="logwages"`, earnings play that role, as in Blundell, Chen, and Kristensen
(2007). If earnings shaped the food share in other ways, the curve would absorb
those effects as well.
[Total spending as its own instrument](#total-spending-as-its-own-instrument) tries
the alternative, `wname="logexp"`, to show what the instrument changes.

:::{admonition} Pass one column twice for a regression
:class: important

When `xname` and `wname` hold the same values, {func}`~moderndid.npiv` uses the
basis for spending as its own instrument basis. Every fit is then an ordinary
least squares regression of the food share on the splines.
:::

### Cubic splines with a data-chosen number of segments

Chen, Christensen, and Kankanala (2024) single out the number of segments as the
key tuning choice. With too few, the splines can't follow the curve and its
estimate is biased. With too many, the estimate turns noisy, since earnings pin
down fine detail only weakly.

With `j_x_degree=3` and `j_x_segments=None`, the curve is a cubic spline whose
number of segments the data choose. The authors' selection first caps the
candidates at what the sample size and the strength of the instrument support. It
then keeps the fewest segments whose curve differs from every finer candidate by no
more than bootstrap noise. The instrument gets quartic splines with `k_w_degree=4`,
since the average of $h_0(X)$ given earnings is smoother than $h_0$ itself. Setting
`k_w_smooth=2` gives the instrument four times as many segments as the curve, one
of the two settings the authors recommend.

The main alternative is to fix the number of segments yourself through
`j_x_segments`.
[A number of segments fixed in advance](#a-number-of-segments-fixed-in-advance)
shows what that alternative costs in band width and shape. The other recommended
instrument setting, `k_w_smooth=1`, gets
[its own check](#half-as-many-instrument-segments) as well.

### Bands over the middle of the spending range

The grid and the bands settle where to read the curve and how much to trust it
there. Because only 24 households sit beyond each end of the range from 4.5 to 6.5,
we evaluate the curve at 81 points between those values with `x_eval=grid`.
Passing the same points as `x_grid` makes the selection compare its candidates
there too, where the data are dense.
[Selection over the full spending range](#selection-over-the-full-spending-range)
shows what happens when `x_grid` is left out.

With `alpha=0.05`, each band is built so that the chance it misses its function
anywhere on the grid is at most 5 percent. If a 10 percent chance is acceptable for
your purpose, `alpha=0.10` narrows both bands. The 1,000 bootstrap draws in
`biters=1000` match the number Chen, Christensen, and Kankanala (2024) use.
[Ten bootstrap seeds](#ten-bootstrap-seeds) swaps `seed=7` for other seeds to see
how much the selection and the bands move.

:::{admonition} Seed npiv with seed, not random_state
:class: tip

Where most moderndid estimators take `random_state` for their bootstrap draws,
{func}`~moderndid.npiv` takes `seed`. Since both the selection and the bands draw
from it, a call without it gives new critical values on every run.
:::

One call to {func}`~moderndid.npiv` picks the number of segments, fits the curve,
and builds both bands.

```{code-cell} ipython3
# Estimate the curve and its slope along with their bands.
result = did.npiv(data, **spec)

# The number of segments the data chose, out of the candidates the selection compared.
candidates = result.args["j_x_segments_set"].tolist()
print(f"segments chosen {result.j_x_segments} from the candidates {candidates}")
print(f"basis functions for the curve {result.args['j_tilde']}")
print(f"critical values {result.cv:.4f} for the curve and {result.cv_deriv:.4f} for the slope")
```

## The food share across total spending

The selection settled on a single segment out of the candidates 1, 2, 4, 8, and
16. The curve is therefore one cubic polynomial in log spending, built from 4 basis
functions. Unlike a line or a quadratic fixed in advance, that single cubic piece is
what the data picked over finer splines with 2 to 16 segments. Its critical values,
3.8678 for the curve and 3.8263 for the slope, sit well above the 1.96 of a
pointwise interval, in part because each band covers every point of its function
at once.

:::{admonition} A margin for the data's choice
:class: note

The data-driven critical value takes its bootstrap quantile over several of the
candidates and adds a term from the selection step. That margin covers
approximation bias that a number of segments picked from the data can't rule out.
[A number of segments fixed in advance](#a-number-of-segments-fixed-in-advance)
measures how much width it adds.
:::

The result keeps the curve in `h` and its band in `h_lower` and `h_upper`. The
slope and its band sit in `deriv`, `h_lower_deriv`, and `h_upper_deriv`. The cell
below prints all six at every quarter step of log spending.

```{code-cell} ipython3
# The curve and its slope with their bands at every quarter step of log total spending.
print(f"{'log spending':>12}{'food share':>12}{'95% band':>20}{'slope':>10}{'95% band':>21}")
for i in range(0, len(grid), 10):
    share_band = f"[{result.h_lower[i]:.4f}, {result.h_upper[i]:.4f}]"
    slope_band = f"[{result.h_lower_deriv[i]:7.4f}, {result.h_upper_deriv[i]:7.4f}]"
    row = f"{grid[i]:>12.2f}{result.h[i]:>12.4f}{share_band:>20}"
    print(f"{row}{result.deriv[i]:>10.4f}{slope_band:>21}")

# A flat line fits inside the band only if the lowest upper edge clears the highest lower edge.
highest_lower = result.h_lower.max()
lowest_upper = result.h_upper.min()
print(f"\nhighest lower edge {highest_lower:.4f}, lowest upper edge {lowest_upper:.4f}")
print(f"the curve falls at {np.sum(np.diff(result.h) < 0)} of {len(grid) - 1} steps")
```

You can see the food share fall at all 80 steps of the grid, from 0.2614 at log
spending 4.5 to 0.1288 at 6.5. Households at the top of the range devote about half
as large a share of their budget to food as those at the bottom.

The band then settles whether the share could stay flat as spending rises. A flat
line inside it would have to sit above the band's highest lower edge of 0.2105 and
below its lowest upper edge of 0.1810 at once. Since no line can do both, the band
rules out a constant food share. It's narrowest near the middle of the grid, 0.0338
wide at 5.5, and widens to 0.1580 at 4.5 and 0.1073 at 6.5, where fewer households
pin the curve down.

In the plot, you can check by eye that no horizontal line fits between the dashed
edges of the shaded band around the curve.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: The food share falling from 0.26 to 0.13 across log total spending inside its band
---
import polars as pl
from plotnine import aes, geom_line, geom_ribbon, ggplot, labs

# The curve and its band, drawn the way moderndid draws a dose-response curve.
curve = pl.DataFrame(
    {"spending": grid, "share": result.h, "lower": result.h_lower, "upper": result.h_upper}
)
(
    ggplot(curve, aes("spending", "share"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#5b7ea4", alpha=0.2)
    + geom_line(aes(y="lower"), linetype="dashed", color="#2c3e50", size=0.5)
    + geom_line(aes(y="upper"), linetype="dashed", color="#2c3e50", size=0.5)
    + geom_line(color="#2c3e50", size=1)
    + labs(x="Log total spending", y="Food share")
    + did.theme_moderndid()
)
```

## How fast the share falls

In the slope columns of the same table, you can read how steeply the share falls
at each level of spending. The point estimates stay between −0.0555 and −0.0566 up
to log spending 5.0 and then steepen to −0.0934 at 6.5. Among households that
already spend a lot, the share falls faster with each further rise in spending.

Dividing each slope by the share and adding 1 gives the elasticity of food
spending at that point. It comes to 0.760 at log spending 5.0, 0.697 at 5.5, and
0.564 at 6.0. A household at 5.5 that spends 10 percent more in total spends about
7 percent more on food. That's what Engel's law means when it calls food a
necessity.

The slope's own band asks more of the data, since it has to sit below zero at a
point before it shows the share falling there. The next cell finds where that band
lies entirely below zero.

```{code-cell} ipython3
# The points of the grid where the whole slope band lies below zero.
below = grid[result.h_upper_deriv < 0]
print(f"slope band below zero at {len(below)} of {len(grid)} points")
print(f"from log spending {below.min():.3f} to {below.max():.3f}")
```

The band lies wholly below zero at only 15 of the 81 points, from log spending
5.750 to 6.100. Everywhere else it reaches above zero, even though the band for
the curve rules out a flat line. Ruling out a flat curve only takes a difference
in the share between two points of the grid, a much weaker claim than a negative
slope at every point.

:::{admonition} Elasticities carry no band of their own
:class: warning

The elasticities above are point estimates, since {func}`~moderndid.npiv` builds
bands for the curve and its slope but not for their ratio. The slope band confirms
that food is a necessity only between log spending 5.750 and 6.100.
:::

In the plot of the slope, the band dips wholly below the gray zero line only on
that short stretch near 6.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: The slope of the food share with a band that lies below zero only from 5.75 to 6.1
---
from plotnine import geom_hline

# The slope and its band, with a line at zero.
slope = pl.DataFrame(
    {
        "spending": grid,
        "slope": result.deriv,
        "lower": result.h_lower_deriv,
        "upper": result.h_upper_deriv,
    }
)
(
    ggplot(slope, aes("spending", "slope"))
    + geom_hline(yintercept=0, color="#7f8c8d")
    + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="#5b7ea4", alpha=0.2)
    + geom_line(aes(y="lower"), linetype="dashed", color="#2c3e50", size=0.5)
    + geom_line(aes(y="upper"), linetype="dashed", color="#2c3e50", size=0.5)
    + geom_line(color="#2c3e50", size=1)
    + labs(x="Log total spending", y="Slope of the food share")
    + did.theme_moderndid()
)
```

## How much each choice matters

The five checks below each change one choice that could move the curve or its band.

### Total spending as its own instrument

To measure what the instrument does to the curve, the cell below refits it with
`wname="logexp"`.

```{code-cell} ipython3
# Use total spending as its own instrument to turn the fit into a regression.
regression = did.npiv(data, **(spec | {"wname": "logexp"}))
fits = {"earnings as instrument": result, "regression": regression}

# Each fit's segments, food share at 5.0 and 6.0, the fall between them, and average band width.
print(f"{'':<24}{'segments':>9}{'share 5.0':>11}{'share 6.0':>11}{'fall':>8}{'band width':>12}")
for name, fit in fits.items():
    at_5, at_6 = np.interp([5.0, 6.0], grid, fit.h)
    width = np.mean(fit.h_upper - fit.h_lower)
    shares = f"{at_5:>11.4f}{at_6:>11.4f}{at_5 - at_6:>8.4f}"
    print(f"{name:<24}{fit.j_x_segments:>9}{shares}{width:>12.4f}")

# The grid points where the regression's curve lies inside the band of the instrumented curve.
inside_band = (regression.h >= result.h_lower) & (regression.h <= result.h_upper)
print(f"\nregression curve inside our band at {inside_band.sum()} of {len(grid)} points")
print(f"outside it at log spending {grid[~inside_band].round(3).tolist()}")
```

Although the regression also settles on a single segment, its curve falls by
0.1113 between log spending 5.0 and 6.0, about 1.8 times the 0.0630 of the
instrumented curve. Its band averages 0.0349 across the grid against 0.0632, since
a regression uses all the variation in spending instead of only the part that
earnings explain.

Since the band of the instrumented curve is built to hold the true curve at every
point at once, the regression's curve leaving it at 6.2, 6.225, and 6.25 counts
against the regression, if only barely. That verdict rests on an instrument whose
validity the data can't check. Preferring the instrumented curve comes down to
believing that earnings are unrelated to food tastes and affect the food share only
through total spending.

The plot draws both curves with their bands, the instrumented curve in dark blue
and the regression in gray.

```{code-cell} ipython3
---
tags: [hide-input]
mystnb:
  image:
    alt: The instrumented curve and the steeper regression curve with their bands
---
from plotnine import scale_color_manual, scale_fill_manual

# Stack both fits and their bands, labeled by how each treats total spending.
both = pl.concat(
    pl.DataFrame(
        {"spending": grid, "share": fit.h, "lower": fit.h_lower, "upper": fit.h_upper, "fit": name}
    )
    for name, fit in fits.items()
)
(
    ggplot(both, aes("spending", "share", color="fit", fill="fit"))
    + geom_ribbon(aes(ymin="lower", ymax="upper"), alpha=0.25, color="none")
    + geom_line(size=1)
    + scale_color_manual(values={"earnings as instrument": "#2c3e50", "regression": "#7f8c8d"})
    + scale_fill_manual(values={"earnings as instrument": "#5b7ea4", "regression": "#bfbfbf"})
    + labs(x="Log total spending", y="Food share", color="", fill="")
    + did.theme_moderndid()
)
```

### A number of segments fixed in advance

A band for a fixed number of segments is valid only when that number is large
enough for the approximation bias to be negligible. The cell below fixes
`j_x_segments` at each of the first four candidates in turn and compares the
results with the data's choice.

```{code-cell} ipython3
# The same specification with the number of segments fixed instead of chosen.
fixed = {}
print("segments  critical  band width  rising steps  rules out flat")
for segments in (1, 2, 4, 8):
    fit = did.npiv(data, **(spec | {"j_x_segments": segments}))
    fixed[segments] = fit
    width = np.mean(fit.h_upper - fit.h_lower)
    rises = np.sum(np.diff(fit.h) > 0)
    flat_ruled_out = "yes" if fit.h_lower.max() > fit.h_upper.min() else "no"
    print(f"{segments:>8}{fit.cv:>10.4f}{width:>12.4f}{rises:>14}{flat_ruled_out:>16}")
```

Fixing a single segment reproduces our curve exactly, since that's the number the
data chose. Its band, however, averages 0.0427 against 0.0632, about a third
narrower, because its critical value of 2.6169 leaves out the margin for the data's
choice. That narrower band would only be safe if you knew in advance that one
segment was enough.

Fixing more segments, the usual way to keep that bias small, makes the band wider.
Its average width grows from 0.0693 with two segments to 0.1277 with four and
reaches 0.2272 with eight, 3.6 times the width of ours. With two segments the curve
rises on 14 of the 80 steps and with eight on 28. With four it still falls at every
step, as our single segment does. At eight segments the band grows wide enough for
a flat line to fit inside it.

:::{admonition} Let k_w_segments follow the curve
:class: tip

Two-stage least squares needs at least as many instrument functions as spline
functions for spending. Left out, `k_w_segments` follows
`j_x_segments * 2**k_w_smooth`, four times the curve's segments here. A value that
leaves the instrument basis with fewer functions than the spline basis makes
{func}`~moderndid.npiv` raise an error.
:::

### Half as many instrument segments

With `k_w_smooth=1`, the instrument gets twice as many segments as the curve rather
than four times.

```{code-cell} ipython3
# Give the instrument twice as many segments as the curve instead of four times.
coarser = did.npiv(data, **(spec | {"k_w_smooth": 1}))

# The fall from 5.0 to 6.0, the average band width, and the flat-line test.
at_5, at_6 = np.interp([5.0, 6.0], grid, coarser.h)
width = np.mean(coarser.h_upper - coarser.h_lower)
flat_ruled_out = "yes" if coarser.h_lower.max() > coarser.h_upper.min() else "no"

print(f"segments chosen {coarser.j_x_segments}, instrument segments {coarser.k_w_segments}")
print(f"share at 5.0 {at_5:.4f}, at 6.0 {at_6:.4f}, fall {at_5 - at_6:.4f}")
print(f"critical value {coarser.cv:.4f}, band width {width:.4f}, rules out flat {flat_ruled_out}")
```

The selection again picks one segment, now with 2 instrument segments instead of
4. The curve comes out flatter and falls by 0.0432 between 5.0 and 6.0 rather than
0.0630. Its band widens to an average of 0.0838, though it still rules out a flat
line.

### Selection over the full spending range

Leaving `x_grid` out lets the selection compare its candidates over the full range
of spending, from 3.61 to 7.43. The cell makes that change for the instrumented
curve and for the regression of the first check.

```{code-cell} ipython3
# The selection compared over the full range of total spending, for both fits.
full_range = did.npiv(data, **(spec | {"x_grid": None}))
full_range_regression = did.npiv(data, **(spec | {"wname": "logexp", "x_grid": None}))
full_range_fits = {"earnings as instrument": full_range, "regression": full_range_regression}
for name, fit in full_range_fits.items():
    searched = fit.args["j_x_segments_set"].tolist()
    print(f"{name:<24}candidates {searched}, chose {fit.j_x_segments}, critical value {fit.cv:.4f}")

# How many of 64 equal intervals over the full range hold no household.
counts, _ = np.histogram(spending, bins=64)
print(f"{np.sum(counts == 0)} of 64 equal intervals over the full range hold no household")
```

For the instrumented curve, the full range leaves the single segment in place and
moves the critical value only from 3.8678 to 3.9114. Because no instrument limits a
regression's candidates, they run all the way to 128 segments. Over the full range
the regression settles on 64 segments, so many that 12 of 64 equal intervals across
the range hold no household at all.

:::{admonition} Thin tails can inflate the segment count
:class: warning

When a regression's selection compares candidates in sparse tails, it can settle
on far more segments than it would where the data are dense. An `x_grid` over the
dense part of the data, as in `spec`, guards against this.
:::

### Ten bootstrap seeds

Since both the selection and the critical values come from bootstrap draws, a
different seed could in principle change either one.

```{code-cell} ipython3
# The main specification under the ten bootstrap seeds from 0 to 9.
print(f"{'seed':>4}{'segments':>10}{'critical value':>16}")
for seed in range(10):
    fit = did.npiv(data, **(spec | {"seed": seed}))
    print(f"{seed:>4}{fit.j_x_segments:>10}{fit.cv:>16.4f}")
```

All ten seeds pick a single segment and so give the same curve. Across the ten,
the critical value ranges from 3.7475 to 3.8678. Since our seed of 7 gives the
largest of them, the curve's band on this page is the widest that any of the ten
seeds would draw.

### Comparing the checks

The table below collects each check's number of segments, the fall in the food
share from 5.0 to 6.0, the average band width, and whether the band rules out a
flat line.

```{code-cell} ipython3
:tags: [hide-input]

# Each check's segments, the fall in the food share from 5.0 to 6.0, its band width, and the
# flat-line test.
checks = {
    "our specification": result,
    "regression": regression,
    "1 segment fixed": fixed[1],
    "2 segments fixed": fixed[2],
    "4 segments fixed": fixed[4],
    "8 segments fixed": fixed[8],
    "instrument segments halved": coarser,
    "selection over full range": full_range,
}

print(f"{'check':<27}{'segments':>9}{'fall 5 to 6':>13}{'band width':>12}{'rules out flat':>16}")
for name, fit in checks.items():
    at_5, at_6 = np.interp([5.0, 6.0], grid, fit.h)
    width = np.mean(fit.h_upper - fit.h_lower)
    flat_ruled_out = "yes" if fit.h_lower.max() > fit.h_upper.min() else "no"
    print(f"{name:<27}{fit.j_x_segments:>9}{at_5 - at_6:>13.4f}{width:>12.4f}{flat_ruled_out:>16}")
```

The table sorts the choices into those that move the fall in the food share and
those that mostly change how wide its band is. Dropping the instrument changes the
fall the most, since the regression's curve falls by 0.1113 between 5.0 and 6.0
against 0.0630 for ours. Half as many instrument segments come next and flatten the
fall to 0.0432. Fixed numbers of segments keep the fall between 0.0630 and 0.0713
but change the band from about a third narrower with one segment to 3.6 times as
wide with eight. Selecting over the full range and changing the seed leave the
curve alone.

In every row of the table except eight fixed segments, the band rules out a food
share that stays flat as total spending rises. How fast it falls depends most on
whether earnings serve as the instrument and on how finely the instrument's splines
are cut. Only the first of those choices leans on the assumption that earnings are
unrelated to food tastes and reach the food share only through total spending. With
the instrument set equal to the regressor, the same estimator gives the
dose-response curve that the {ref}`continuous treatment example <example_cont_did>`
estimates with `dose_est_method="cck"`, where the curve is an effect at each
dose. The [background page on nonparametric instrumental variables](../background/npiv)
writes out the selection rule and the margin in the data-driven band.
