<img src="./docs/source/_static/drsynthdid.png" width="175" align="left" alt="drsynthdid logo"></img>

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Build Status](https://github.com/jordandeklerk/drdidsynth/actions/workflows/test.yml/badge.svg)](https://github.com/jordandeklerk/drdidsynth/actions/workflows/test.yml)
[![Last commit](https://img.shields.io/github/last-commit/jordandeklerk/drdidsynth)](https://github.com/jordandeklerk/drdidsynth/graphs/commit-activity)
[![Commit activity](https://img.shields.io/github/commit-activity/m/jordandeklerk/drdidsynth)](https://github.com/jordandeklerk/drdidsynth/graphs/commit-activity)
[![Python version](https://img.shields.io/badge/3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?logo=python&logoColor=white)](https://www.python.org/)

__DR-synthdid__ is a Python package that implements a doubly robust methodology for causal inference in panel data, combining Difference-in-Differences and Synthetic Control approaches. It provides semiparametric estimation of average treatment effects that remain valid under either parallel trends or synthetic control assumptions, with built-in support for multiplier bootstrap inference, repeated cross-sectional data, and staggered treatment designs.

__DR-synthdid__ is an unofficial implementation of the paper [Difference-in-Differences Meets Synthetic Control: Doubly Robust Identification and Estimation](https://arxiv.org/pdf/2503.11375).


> **⚠️ Note:**
> This package is currently in development.



## Citation

```bibtex
@misc{sun2025difference,
  title        = {Difference-in-Differences Meets Synthetic Control: Doubly Robust Identification and Estimation},
  author       = {Sun, Yixiao and Xie, Haitian and Zhang, Yuhang},
  howpublished = {arXiv preprint arXiv:2503.11375},
  year         = {2025},
  doi          = {10.48550/arXiv.2503.11375},
  url          = {https://arxiv.org/abs/2503.11375}
}
```
