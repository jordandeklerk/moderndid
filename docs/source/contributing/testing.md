# Testing DR-synthdid

## How to run the test suite
The recommended way to run the test suite is to do it via `tox`.
Tox manages the environment, its env variables and the command to run
to allow testing the library with different combinations of optional dependencies
from the same development env.

To run the full test suite run:

```bash
tox -e py312  # or py310, py311, py313, should match your python version
```

To run only the minimal parts of the test suite:

```bash
tox -e minimal  # should work for any python version
```

To run style checks:

```bash
tox -e check
```

## How to write tests

Use the `importorskip` helper function from `tests/helpers.py` for any import outside of
the Python standard library plus NumPy. For example:

```python
import numpy as np

from .helpers import importorskip

pd = importorskip("pandas")

#... in the code use pd.DataFrame, pd.Series as usual
```

As `importorskip` will skip all tests in that file, tests should be divided into
files with tests of the core functionality always being in their own file
with no optional dependencies import, and tests that require optional dependencies
in a separate file.

## About DR-synthdid testing

The test suite is structured to ensure both the core functionality and the optional
dependencies integrations work as expected.

The `importorskip` helper function from `tests/helpers.py` is used when importing
optional dependencies so that tests are skipped if a dependency is not available.
In addition, the env variable `DRSYNTHD_REQUIRE_ALL_DEPS` can be set to disable this behavior
and ensure uninstalled dependencies raise an error.

When using `tox -e pyXXX` all optional dependencies are installed,
and `DRSYNTHD_REQUIRE_ALL_DEPS` is set to ensure all tests in the test suite run.
However, `tox -e minimal` only installs the core dependencies and doesn't set the env variable,
which ensures that the minimal install is viable and works as expected.

On GitHub Actions, the full test suite is run for all supported Python versions
and the minimal test suite for one Python version.
The test configuration is defined by the combination of `tox.ini` and `.github/workflows/test.yml`.

## Data Generation for Testing

DR-synthdid includes data generation utilities specifically designed for testing:

- **DiDDGP**: Data generating process for difference-in-differences designs. Creates synthetic panel data with treatment effects for DiD analysis.
- **SyntheticControlsDGP**: Data generating process for synthetic control designs. Creates synthetic panel data where one or more units are treated after a pre-treatment period.

These data generating process (DGP) classes create synthetic data that can be used
for testing both Difference-in-Differences (DiD) and Synthetic Controls methods.

The implementation can be found in the `tests/dgp.py` file in the repository.
