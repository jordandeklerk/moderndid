---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

(scaling)=

# Scaling your analysis

Because a staggered analysis fits a separate comparison for each treatment
cohort and period, its running time depends on both the size of the data and
the number of comparisons. Bootstrap inference adds another calculation that
can take time even after those comparisons have been fitted. We can speed up
some of this work on one machine by fitting comparisons in parallel, using
compiled CPU helpers, or moving supported calculations to an NVIDIA GPU.

The choice depends on where your analysis spends its time. More worker threads
can help when you have many comparisons to fit, whereas a GPU may help with
large matrix calculations within a comparison. Neither setting changes the
identification assumptions or the effect you have chosen to estimate.

```{toctree}
:hidden:
:maxdepth: 1

gpu
```

## Fitting comparisons in parallel

When you have many group-time comparisons to fit, {func}`~moderndid.att_gt`
and {func}`~moderndid.ddd` can work on them in parallel. Their `n_jobs` argument
controls the number of worker threads, with the default `n_jobs=1` fitting
the comparisons sequentially. You can choose a positive integer for a fixed
number of threads or use `n_jobs=-1` for the machine's reported CPU count.
These threads share the data within one Python process rather than copying
it to separate worker processes.

We will use the minimum wage data to show the change at the call site. The
specification below keeps the same comparison group, covariates, and inference
settings in both fits so you can compare their estimated effects directly.

```{code-cell} ipython3
import moderndid as did
import numpy as np

data = did.load_mpdta()
spec = {
    "data": data,
    "yname": "lemp",
    "tname": "year",
    "idname": "countyreal",
    "gname": "first.treat",
    "xformla": "~ lpop",
    "control_group": "nevertreated",
    "est_method": "dr",
    "base_period": "universal",
    "boot": True,
    "cband": True,
    "biters": 999,
    "random_state": 42,
}

sequential = did.att_gt(**spec, n_jobs=1)
parallel = did.att_gt(**spec, n_jobs=2)

np.testing.assert_allclose(sequential.att_gt, parallel.att_gt)
print(parallel)
```

The assertion checks that the treatment effect estimates agree within numerical
tolerance. Since you still receive the usual result object, aggregation and
plotting use the same functions as the {ref}`staggered example <example_staggered_did>`.
This small dataset demonstrates the setting; it does
not establish that using two threads will make your own analysis faster.

{func}`~moderndid.didml` also parallelizes group-time comparisons through
`n_jobs`. For {func}`~moderndid.diddynamic.dyn_balancing`, that argument applies when you
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

## Using compiled CPU helpers

If bootstrap or aggregation calculations take much of the running time, you
can install the optional Numba dependency for the CPU helpers that support it.
ModernDiD selects those compiled helpers automatically when Numba is available
without requiring another argument in your estimator call.

```bash
python -m pip install "moderndid[numba]"
```

The first call to a compiled helper may include compilation time. When you
compare repeated analyses, run the same specification once before timing it
so that you can distinguish this initial cost from the time later fits take.
Measure the complete analysis to see whether Numba helps your workload because
it does not compile every part of an estimator.

## Moving numerical work to a GPU

The CuPy backend moves supported numerical operations to an NVIDIA GPU while
data preparation and orchestration still run on the CPU. Larger matrix
calculations may justify the transfers between CPU and GPU memory; many small
comparisons may spend more time on those transfers than they save. If your
analysis spends time on these larger calculations, the {doc}`gpu` guide shows
you how to check the CUDA environment and choose which calls use the GPU.

Local threading and GPU acceleration both operate within one machine rather
than distributing an analysis across a cluster. The current package provides
neither a Dask nor a Spark estimation backend for these calculations.

```{eval-rst}
.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: section-index-table

   * - Guide
     - What you will learn
   * - :doc:`gpu`
     - Set up CuPy, select supported GPU calculations, and measure their running time.
```

<p class="mdid-footer-logo"><img src="../_static/logo-wordmark.svg" alt="ModernDiD logo"></p>
