.. _benchmarking:

=============
Benchmarking
=============

**ModernDiD** includes a benchmark suite that measures the computational
performance of Python estimators against their canonical R implementations.
You can run predefined suites out of the box, write custom configurations,
and add benchmarks for new estimators.

Running benchmarks
==================

Quick start
-----------

The fastest way to run benchmarks is with a predefined suite using the
``--python-only`` flag (skips R, which requires the R packages to be
installed)::

   python -m benchmark.run_benchmark attgt --suite quick --python-only
   python -m benchmark.run_benchmark ddd --suite quick --python-only
   python -m benchmark.run_benchmark contdid --suite quick --python-only
   python -m benchmark.run_benchmark didinter --suite quick --python-only

You can also run the module-specific entry points directly::

   python -m benchmark.did.run_benchmark --suite quick --python-only
   python -m benchmark.didtriple.run_benchmark --suite quick --python-only
   python -m benchmark.didcont.run_benchmark --suite quick --python-only
   python -m benchmark.didinter.run_benchmark --suite quick --python-only

Available estimators
--------------------

The benchmark CLI has four subcommands, one per estimator.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Subcommand
     - **ModernDiD** function
     - R package comparison
   * - ``attgt``
     - ``att_gt``
     - R ``did``
   * - ``ddd``
     - ``ddd``
     - R ``triplediff``
   * - ``contdid``
     - ``cont_did``
     - R ``contdid``
   * - ``didinter``
     - ``did_multiplegt``
     - R ``DIDmultiplegtDYN``

Predefined suites
-----------------

Each estimator has predefined benchmark suites that test different scaling
dimensions. For the ``attgt`` estimator, for example, the available suites
are listed below.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Suite name
     - What it tests
   * - ``quick``
     - Small-scale sanity check (100 to 1,000 units)
   * - ``scaling_units``
     - Scaling from 100 to 100,000 units
   * - ``scaling_periods``
     - Scaling from 5 to 20 time periods
   * - ``scaling_groups``
     - Scaling from 3 to 10 treatment groups
   * - ``est_methods``
     - Comparing DR, IPW, and regression estimators
   * - ``bootstrap``
     - Scaling with bootstrap iterations (100 to 1,000)
   * - ``large_scale``
     - Stress test (100,000 to 2,000,000 units)

Run a specific suite::

   python -m benchmark.run_benchmark attgt --suite scaling_units

Custom configurations
---------------------

For one-off benchmarks, pass parameters directly instead of using a suite::

   python -m benchmark.run_benchmark attgt \
       --n-units 5000 \
       --n-periods 10 \
       --n-groups 5 \
       --est-method dr \
       --warmup 2 \
       --runs 10 \
       --seed 42

The following common parameters are shared across all estimators.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--warmup``
     - 1
     - Number of warmup runs (not timed, primes caches)
   * - ``--runs``
     - 5
     - Number of timed runs (results are averaged)
   * - ``--seed``
     - 42
     - Random seed for data generation (reproducibility)
   * - ``--python-only``
     - false
     - Skip R benchmarks
   * - ``--output-dir``
     - ``benchmark/output``
     - Directory for results and plots
   * - ``--quiet``
     - false
     - Suppress verbose output
   * - ``--boot``
     - false
     - Enable bootstrap inference
   * - ``--biters``
     - 100
     - Number of bootstrap iterations

Including R comparisons
-----------------------

To run benchmarks that compare against R, you need the corresponding R
packages installed. The same R packages used by the validation test suite
work here. If you've already run ``pixi run -e validation setup-r``, you're
set.

Without the ``--python-only`` flag, each benchmark configuration runs the
following steps.

1. Generates a synthetic dataset in Python
2. Exports the dataset to CSV for R
3. Runs the Python estimator with warmup + timed runs
4. Runs the R estimator via ``subprocess`` with the same protocol
5. Reports timing comparisons

Interpreting results
====================

The benchmark suite produces three types of output. The console summary
shows median and mean runtimes for each configuration, with speedup ratios
when R benchmarks are included. Saved results are JSON files in the output
directory with full timing data. Plots are PNG files showing scaling
behavior across configurations.

When interpreting results, keep the following in mind.

- Focus on medians, not means. A single slow run (e.g., due to garbage
  collection or OS scheduling) can skew the mean.
- Warmup runs matter. The first run is often slower due to JIT compilation
  (Numba), import overhead, and cache cold-starts. The benchmark suite
  handles this automatically, but if you're timing manually, always include
  warmup.
- Compare relative scaling. Absolute runtimes depend on hardware. The
  interesting question is usually how performance scales with dataset size,
  not the raw seconds.
- R comparisons use subprocess. The R timing includes R startup overhead,
  CSV parsing, and package loading. For very small datasets, this overhead
  may dominate the actual computation time, making Python appear faster than
  it really is for the statistical computation alone.

Benchmark structure
===================

The benchmark code lives in `benchmark <https://github.com/jordandeklerk/moderndid/tree/main/benchmark>`__ and mirrors the package's module
structure::

   benchmark/
   ├── run_benchmark.py       # Unified CLI entry point
   ├── common/                # Shared utilities
   ├── did/                   # att_gt benchmarks
   │   ├── config.py          # ATTgtBenchmarkConfig dataclass and suites
   │   ├── dgp.py             # Data generation
   │   ├── runners.py         # Python and R runner functions
   │   ├── storage.py         # Result serialization
   │   └── run_benchmark.py   # Module-specific CLI
   ├── didcont/               # cont_did benchmarks
   ├── didinter/              # did_multiplegt benchmarks
   ├── didtriple/             # ddd benchmarks
   ├── output/                # Generated results and plots
   └── plot.py                # Cross-estimator plotting

Each estimator module follows the same pattern. ``config.py`` defines a
``@dataclass`` with benchmark parameters and a dictionary of named suites.
``dgp.py`` generates synthetic data matching the estimator's expected input
format. ``runners.py`` contains ``run_python()`` and ``run_r()`` functions
that execute the estimator and return timing results. ``run_benchmark.py``
wires everything together with ``argparse``.

Adding a new benchmark
======================

To add benchmarks for a new estimator, follow the steps below.

1. **Create the directory**::

      mkdir benchmark/newestimator

2. **Define the config** in ``config.py``. Follow the existing pattern with
   a dataclass and a ``SUITES`` dictionary::

      @dataclass
      class NewEstimatorBenchmarkConfig:
          n_units: int = 1000
          n_periods: int = 5
          # ... estimator-specific parameters
          n_warmup: int = 1
          n_runs: int = 5
          random_seed: int = 42

      NEWESTIMATOR_BENCHMARK_SUITES = {
          "quick": [
              NewEstimatorBenchmarkConfig(n_units=100),
              NewEstimatorBenchmarkConfig(n_units=500),
              NewEstimatorBenchmarkConfig(n_units=1000),
          ],
          "scaling_units": [
              # progressively larger datasets
          ],
      }

3. **Implement the data generator** in ``dgp.py``. Use the project's data
   generation functions from ``moderndid.core.data`` where possible.

4. **Implement runners** in ``runners.py``. The Python runner should call the
   estimator and time it. The R runner should export data to CSV, call R via
   ``subprocess``, and parse the timing output.

5. **Register the subcommand** in `run_benchmark.py <https://github.com/jordandeklerk/moderndid/tree/main/benchmark/run_benchmark.py>`__ by adding a
   new subparser and wiring it to your module's ``main()`` function.

6. **Add a ``quick`` suite** at minimum so others can verify the benchmark
   works without waiting for large-scale runs.
