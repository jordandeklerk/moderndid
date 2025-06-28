============
Installation
============

pyDiD requires Python 3.8 or later.

Installing from PyPI
--------------------

pyDiD can be installed from PyPI using pip:

.. code-block:: bash

    pip install pydid

Installing from source
----------------------

To install the latest development version from GitHub:

.. code-block:: bash

    pip install git+https://github.com/jordandeklerk/pyDiD

Dependencies
------------

Required dependencies:

- numpy (>=1.22.0)
- scipy (>=1.10.0,<1.16)
- pandas (>=2.0.0)
- formulaic (>=0.6.0)
- statsmodels (>=0.14.4)
- scikit-learn (>=1.6.1)
- cvxpy[ECOS] (>=1.3.0)
- sympy (>=1.14.0)

Development installation
------------------------

To install pyDiD for development:

.. code-block:: bash

    git clone https://github.com/jordandeklerk/pyDiD.git
    cd pyDiD
    pip install -e ".[dev]"

This will install pyDiD in editable mode along with all development dependencies.
