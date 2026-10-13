# Contributing to ModernDiD

Thank you for your interest in contributing to ModernDiD!

The [Contributing guide](https://moderndid.readthedocs.io/en/latest/contributing/index.html) covers the development setup, the Git workflow, testing, review, and releases. The [Development guide](https://moderndid.readthedocs.io/en/latest/dev/index.html) explains the library's architecture and how to add an estimator.

## Quick start

The project uses [Pixi](https://pixi.sh/) to manage its development, documentation, and validation environments. The [setup guide](https://moderndid.readthedocs.io/en/latest/contributing/guide.html) explains each step below in more detail.

1. Fork the repository, clone your fork, and add the main repository as `upstream`.
2. Install the development environment with `pixi install -e dev`.
3. Create a branch for your change because the pre-commit hooks refuse commits to `main`.
4. Run the tests that cover your change, as in `pixi run -e dev pytest tests/did/test_att_gt.py -m "not slow" -vv`.
5. Run the pre-commit hooks with `pixi run lint` and resolve what they report.

If you work on Windows or prefer a virtual environment, use Python 3.12 or newer, install the development extras with `python -m pip install -e ".[all,test,dev]"`, and run `pytest` and `pre-commit` directly. The [testing guide](https://moderndid.readthedocs.io/en/latest/contributing/testing.html#understanding-automated-checks) describes how the pull request checks use tox.

For questions or issues, please open a [GitHub issue](https://github.com/jordandeklerk/moderndid/issues).
