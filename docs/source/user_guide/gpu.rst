.. _gpu:

GPU acceleration
================

If large matrix calculations account for much of your analysis time, an
NVIDIA GPU may reduce the time spent fitting models and drawing bootstrap
replications. ModernDiD uses CuPy for supported GPU calculations while keeping
the data and returned results in the forms you already use. We start with a
small estimation to check that the CUDA environment works before measuring
whether the GPU helps with your specification.

The amount of work inside each comparison matters as much as the total number
of rows. A GPU can be slower than the CPU for small comparisons because
transferring arrays and starting GPU kernels also takes time.

Setting up a CUDA environment
-----------------------------

You need an NVIDIA GPU, a driver that supports CUDA 12 or later, and CuPy.
The ``gpu`` extra installs ``cupy-cuda12x[ctk]``, the CUDA 12 build of CuPy
along with the CUDA libraries it needs. Since those libraries come from PyPI
through the ``[ctk]`` extra, the machine needs only the NVIDIA driver rather
than a separate CUDA Toolkit. Check the `CuPy installation guide
<https://docs.cupy.dev/en/stable/install.html>`_ before choosing the extra for
your machine.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add "moderndid[gpu]"

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install "moderndid[gpu]"

A driver that supports CUDA 13 also runs this CUDA 12 build. If you prefer
CuPy's CUDA 13 build, install ModernDiD without the ``gpu`` extra and add
``cupy-cuda13x[ctk]`` beside it.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: console

         uv add moderndid "cupy-cuda13x[ctk]"

   .. tab-item:: pip

      .. code-block:: console

         python -m pip install moderndid "cupy-cuda13x[ctk]"

Since CuPy's wheels conflict with each other, keep only one CuPy distribution
in the environment where you install ModernDiD. On a machine without a
supported NVIDIA GPU, run the Python analysis in a remote GPU environment and
install these dependencies there.

Before fitting a model, check that the backend can use the GPU.

.. code-block:: python

   import moderndid as did

   print(did.HAS_CUPY)

   with did.use_backend("cupy"):
       xp = did.get_backend()
       print(xp.__name__)
       print(float(xp.arange(3).sum()))

The ``HAS_CUPY`` flag reports whether CuPy was importable when ModernDiD
loaded its backend module. Entering the context checks for an available CUDA
device before the sum tests a small GPU calculation that your estimator will
need to perform.
If this setup fails, the :ref:`troubleshooting section <gpu-troubleshooting>`
below explains where to check the installation.

Trying an estimation on the GPU
-------------------------------

:func:`~moderndid.att_gt`, :func:`~moderndid.ddd`, and
:func:`~moderndid.cont_did` accept ``backend="cupy"`` to run a single call on
the GPU. That call temporarily activates CuPy and restores the previous
backend when it returns or raises an exception.

The minimum wage data gives us a small estimation to check the installation.
Its comparison group and covariates also appear in the :ref:`staggered example
<example_staggered_did>`, where you can follow the interpretation of the
estimated effects.

.. code-block:: python

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

   result = did.att_gt(**spec, backend="cupy")
   print(result)

You can pass your usual DataFrame because ModernDiD prepares the data on the
CPU and transfers arrays as the supported calculations need them. Since the
returned result contains CPU arrays, :func:`~moderndid.aggte` and
:func:`~moderndid.plots.plot_gt` use their usual calls.

.. admonition:: Numerical differences between backends
   :class: note

   Small differences in point estimates can arise from numerical rounding.
   Even with the same seed, the backends' random number generators can produce
   different bootstrap draws. CuPy can also change the GPU draws for a seed
   from one release to the next. Fits with nearly collinear covariates depend
   on rounding as well. Their convergence warnings, pseudo-inverse fallbacks,
   and standard errors can therefore differ between the backends.

Choosing which calculations use CuPy
------------------------------------

If several calls should share the GPU backend, use
:func:`~moderndid.use_backend` around that part of your analysis. The context
restores the previous backend on exit so you can keep CPU and GPU work in the
same session without resetting the setting after each call.

.. code-block:: python

   with did.use_backend("cupy"):
       result = did.att_gt(**spec)

   print(did.get_backend().__name__)

You can also use :func:`~moderndid.set_backend` to change the backend in the
current execution context until a later call such as
``did.set_backend("numpy")`` returns it to CPU computation.
Although worker threads created by ModernDiD's ``n_jobs`` setting inherit the
active backend, they all share one GPU.

For staggered DiD and triple differences, supported outcome regressions,
propensity score fits, influence function calculations, and multiplier
bootstrap operations use CuPy. The group-time comparisons and their
scheduling still run from Python on the CPU. Because the two-period
:func:`~moderndid.drdid` and :func:`~moderndid.npiv` functions accept no
``backend`` argument, select CuPy with a context when you want their supported
numerical operations on the GPU.

.. admonition:: Some steps stay on the CPU
   :class: note

   Since :func:`~moderndid.calculate_pscore_ipt` relies on SciPy's optimizer,
   the inverse probability tilting fit that :func:`~moderndid.drdid` uses by
   default runs on the CPU under a CuPy backend. The weighted bootstraps of
   :func:`~moderndid.drdid`, :func:`~moderndid.ipwdid`,
   :func:`~moderndid.ordid`, and :func:`~moderndid.ddd` run on the CPU as well. Because each of their draws
   refits small models, a GPU would save little time there.

The continuous treatment estimator uses CuPy in spline calculations and
bootstrap operations. Its ``dose_est_method="cck"`` path also uses GPU
regression and derivative calculations through the NPIV implementation.
The parametric path mixes CPU and GPU calculations because it transfers
spline bases back to NumPy for its least squares fits.

The intertemporal estimator :func:`~moderndid.did_multiplegt`, dynamic
balancing :func:`~moderndid.diddynamic.dyn_balancing`, machine learning
:func:`~moderndid.didml`, and sensitivity analysis
:func:`~moderndid.honest_did` do not provide a CuPy estimation path.
Selecting a GPU backend for one of the supported estimators does not move
every other part of your analysis to the GPU.

.. admonition:: Choose ETWFE's backend separately
   :class: tip

   :func:`~moderndid.etwfe` selects the backend for fixed effects absorption
   through its own ``backend`` argument. Its ``"cupy"``, ``"cupy32"``,
   ``"cupy64"``, and ``"jax"`` options can use GPU calculations when the
   required libraries and hardware are available. Because this setting is
   independent of :func:`~moderndid.use_backend`, check the ETWFE API when
   choosing it.

Measuring the time your analysis takes
--------------------------------------

Once the installation works, we can measure the complete estimator call so
that preparation and data transfers count toward its running time. Keep the
data, estimation method, clustering, bootstrap iterations, and other
statistical choices fixed when comparing backends. Since CuPy runs GPU
operations asynchronously, synchronize the device before and after the timed
call as described in its `performance guide
<https://docs.cupy.dev/en/stable/user_guide/performance.html>`_.

.. code-block:: python

   import time

   import cupy as cp

   did.att_gt(**spec, backend="cupy")
   cp.cuda.runtime.deviceSynchronize()

   start = time.perf_counter()
   result = did.att_gt(**spec, backend="cupy")
   cp.cuda.runtime.deviceSynchronize()
   elapsed = time.perf_counter() - start

   print(f"GPU estimation took {elapsed:.3f} seconds")

The untimed call allows CUDA initialization and kernel compilation to finish
before the measurement. When you repeat the measurement on your data and
compare it with the same specification under ``backend="numpy"``, check that
the point estimates agree within a suitable numerical tolerance rather than
comparing running time alone. If you also vary ``n_jobs``, keep track of that
setting separately so you can tell which change helped.

Managing GPU memory
-------------------

GPU memory limits the arrays that can participate in a calculation. CuPy
keeps freed allocations in its default memory pool for reuse. Because the pool
holds on to that memory, ``nvidia-smi`` may report it as allocated after a fit
ends. CuPy's `memory management guide
<https://docs.cupy.dev/en/stable/user_guide/memory.html>`_ explains this
behavior and the controls for the pool.

.. admonition:: Leave room for temporary arrays
   :class: tip

   The data's size on disk is not the amount of GPU memory a fit needs.
   Check usage with your full specification before scaling up because design
   matrices, influence functions, and temporary arrays also occupy memory.

The GPU multiplier bootstrap batches its draws to reduce temporary memory
use. If a fit exhausts the GPU, reduce the concurrent work or use
``backend="numpy"``.
Resetting the backend alone does not release memory that the pool has cached.

Although CuPy normally selects device 0, a device context lets you choose
another GPU when your machine has several available. This selects one
device for the call rather than splitting it across those devices.

.. code-block:: python

   with cp.cuda.Device(1):
       result = did.att_gt(**spec, backend="cupy")

.. _gpu-troubleshooting:

Checking installation problems
------------------------------

If ``did.HAS_CUPY`` is false, check that you installed CuPy in the Python
environment running your analysis. Restart the Python process after installing
it because ModernDiD checks import availability when its backend module loads.
An ``ImportError`` from selecting CuPy means that this import check failed.
When CuPy is installed but fails to import, the message repeats CuPy's own
import error rather than reporting CuPy as missing.

A ``RuntimeError`` reporting that no CUDA GPU is available means CuPy imported
but could not use a device. Run ``nvidia-smi`` in that environment and check
that your notebook or remote session has access to a GPU. Driver errors,
missing CUDA headers, and kernel compilation failures need a compatible CUDA
installation even when the CuPy import succeeds.

For details about the installed runtime, run ``cupy.show_config()`` and compare
its report with the `CuPy installation troubleshooting instructions
<https://docs.cupy.dev/en/stable/install.html#faq>`_.
