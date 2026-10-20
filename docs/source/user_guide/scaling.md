---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(scaling)=

# CPU parallelism

An analysis that runs quickly on a small panel can take much longer when you
add counties, treatment cohorts, or observed periods. Staggered DiD fits a
separate comparison for each cohort and period before bootstrap inference
adds its own calculations. We can speed up some of this work on one machine
by fitting comparisons in parallel or by moving supported calculations to an
NVIDIA GPU.

The choice depends on where your analysis spends its time. While more worker
threads can help when you have many comparisons to fit, a GPU may help with
large matrix calculations within a comparison. Neither setting changes the
identification assumptions or the effect you set out to estimate.

## Fitting comparisons in parallel

When you have many group-time comparisons to fit, {func}`~moderndid.att_gt`
and {func}`~moderndid.ddd` can work on them in parallel. Their `n_jobs`
argument sets how many worker threads fit those comparisons. You can keep the
default `n_jobs=1` to fit the comparisons sequentially, choose a positive
integer for a fixed number of threads, or use `n_jobs=-1` for the machine's
reported CPU count. These threads share the data within one Python process
rather than copying it to separate worker processes.

To check what changes when comparisons run in parallel, we fit the minimum
wage data with one worker and two workers. Both calls use the same comparison
group, covariates, and inference settings so their estimated effects can be
compared directly.

```{code-cell} ipython3
import moderndid as did
import numpy as np

# Both fits share the first analysis's specification.
data = did.load_mpdta()
spec = dict(
    yname="lemp",
    tname="year",
    idname="countyreal",
    gname="first.treat",
    control_group="nevertreated",
    xformla="~lpop",
    est_method="dr",
    base_period="universal",
    anticipation=0,
    boot=True,
    biters=10000,
    cband=True,
    random_state=7,
)

# Fit once with one worker thread and once with two.
sequential = did.att_gt(data, **spec, n_jobs=1)
parallel = did.att_gt(data, **spec, n_jobs=2)

# Stop with an error if the two sets of estimates disagree.
np.testing.assert_allclose(sequential.att_gt, parallel.att_gt)
print(parallel)
```

The assertion checks that the treatment effect estimates agree within
numerical tolerance. The returned object still works with the aggregation
and plotting functions used in the
{ref}`staggered DiD example <example_staggered_did>`.
Since this small dataset only demonstrates the setting, time your own
analysis to find out whether two threads also reduce the running time. From a
checkout of the repository, the {ref}`benchmarking guide <benchmarking>` shows
how to time a fit on generated panels with more units, periods, cohorts, or
covariates.

The same `n_jobs` argument parallelizes the group-time comparisons of
{func}`~moderndid.didml` and runs its bootstrap draws in separate worker
processes. Since each worker takes a few seconds to start, a small bootstrap
can run slower with several workers. A script that bootstraps `didml` with
`n_jobs` above 1 must call it inside an `if __name__ == "__main__":` block.

With `boot=True`, {func}`~moderndid.did_multiplegt` reruns the whole estimation
on every bootstrap draw. Its `n_jobs` argument takes the same values as in
`att_gt` and spreads those draws over worker threads. Since the analytic
standard errors need no draws, `n_jobs` has no effect when `boot=False`. For a
given `random_state`, the standard errors stay the same whatever the number of
threads. Memory use does grow with `n_jobs` because each thread holds its own
resampled copy of the panel.

For {func}`~moderndid.diddynamic.dyn_balancing`, the `n_jobs` argument applies when you
request several `histories_length` values or several `final_periods`.
It parallelizes those separate fits rather than the calculations within a
single treatment history. Check the function's signature before passing
`n_jobs` to another estimator because support for the argument varies.

:::{admonition} Start with a few workers
:class: tip

Try two or four threads before using every available CPU core. Since
numerical libraries may also use their own threads, increasing `n_jobs`
can increase contention and temporary memory use enough to slow the fit.
:::

## Moving numerical work to a GPU

The CuPy backend moves supported numerical operations to an NVIDIA GPU while
data preparation and orchestration still run on the CPU. Larger matrix
calculations may justify the transfers between CPU and GPU memory. Many small
comparisons may spend more time on those transfers than they save. If your
analysis spends time on these larger calculations, the
{ref}`GPU acceleration guide <gpu>` shows you how to check the CUDA environment
and choose which calls use the GPU.

Local threading and GPU acceleration both operate within one machine rather
than distributing an analysis across a cluster. When your panel outgrows one
machine or already sits in a Spark table, {func}`~moderndid.att_gt` can fit it
on a Spark cluster instead. The {ref}`Spark guide <spark>` explains when a
Spark fit helps and when an in-memory fit is enough.

<p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
