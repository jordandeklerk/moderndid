# pyDiD

pyDiD is a Python package implementing a variety of DiD and doubly robust DiD estimators with potentially multiple time periods, staggered treatment adoption, and when parallel trends may only be plausible after conditioning on covariates. We also implement an efficient estimator for settings with quasi-random treatment timing and a test for whether parallel trends is insensitive to functional form.

(installation)=
## Installation

Install pyDiD with pip:

::::{tab-set}
:::{tab-item} PyPI
:sync: stable

```bash
pip install pydid
```
:::
:::{tab-item} GitHub
:sync: dev

```bash
pip install git+https://github.com/jordandeklerk/pyDiD
```
:::
::::

```{toctree}
:caption: Reference
:hidden:

background
api/index
```

```{toctree}
:caption: Examples
:hidden:
```

```{toctree}
:caption: Contributing
:hidden:

contributing/testing
```

```{toctree}
:caption: Repository
:hidden:

GitHub repository <https://github.com/jordandeklerk/pyDiD>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
