******************
Project Philosophy
******************

Difference-in-differences has become the leading method for causal inference
in applied empirical research. Over the past decade, econometricians have
developed substantial improvements to the classical two-way fixed effects
approach, addressing problems with staggered adoption, heterogeneous effects,
and sensitivity to parallel trends violations. These methods are now standard
in economics, political science, public health, and business analytics.

Yet Python users have been largely excluded from this methodological progress.
The modern DiD literature has produced excellent R and Stata packages, but
these implementations remain fragmented across dozens of separate libraries,
each with its own API, data requirements, and output formats. Researchers
working in Python have had to either switch languages, write their own
implementations, or use outdated methods that the literature has moved beyond.

ModernDiD exists to bring modern difference-in-differences to Python within
a unified framework that emphasizes performance, consistency, and ease of use.


Why a unified package?
----------------------

The R and Stata ecosystem for DiD is fragmented by design. Each methodological
paper typically produces its own package:
`did <https://bcallaway11.github.io/did/>`_ for
`Callaway and Sant'Anna (2021) <https://arxiv.org/abs/1803.09015>`_,
`DRDID <https://pedrohcgs.github.io/DRDID/>`_ for
`Sant'Anna and Zhao (2020) <https://arxiv.org/abs/1812.01723>`_,
`HonestDiD <https://github.com/asheshrambachan/HonestDiD>`_ for
`Rambachan and Roth (2023) <https://asheshrambachan.github.io/assets/files/hpt-draft.pdf>`_,
`contdid <https://github.com/bcallaway11/contdid>`_ for
`Callaway, Goodman-Bacon, and Sant'Anna (2024) <https://arxiv.org/abs/2107.02637>`_,
and so on. Researchers must learn multiple APIs, convert data between formats,
and reconcile different output structures when combining methods.

ModernDiD consolidates these methods into a single package with a consistent
interface. The same parameter names (``yname``, ``tname``, ``idname``,
``gname``) work across all estimators. Result objects share common structures.
Plotting functions accept outputs from any module. This consistency reduces
the cognitive load of switching between methods and makes it easier to compare
results across different estimation strategies.


Performance
-----------

Causal inference often involves large datasets and computationally intensive
procedures like bootstrapping. ModernDiD is built for performance from the
ground up.

Data wrangling uses `Polars <https://pola.rs/>`_ internally, providing
substantial speed improvements over pandas for the grouping, filtering, and
reshaping operations common in panel data analysis. Users can pass pandas
DataFrames directly and conversion happens automatically.

Numerical computations use NumPy with vectorized operations wherever possible.
Performance-critical inner loops, particularly in bootstrap procedures, use
`Numba <https://numba.pydata.org/>`_ JIT compilation to achieve near-C speeds
while maintaining readable Python code.


Design
------

ModernDiD follows several principles that guide development decisions.

**Correctness.** Every estimator is validated against reference
implementations from the original R packages. Test suites include numerical
comparisons ensuring that ModernDiD produces the same point estimates,
standard errors, and confidence intervals as established implementations.

**Sensible defaults.** The default options reflect best practices from the
methodological literature. Doubly robust estimation is the default because
it provides protection against mis-specification. Never-treated units serve
as the default control group because this avoids contamination from
already-treated units. Users can override these defaults, but the out-of-box
experience should produce credible results.

**Transparency.** Result objects include not just point estimates
but also influence functions, variance-covariance matrices, and estimation
metadata. Users can inspect exactly how estimates were computed and use
intermediate outputs for custom analyses. Warning messages explain when
data issues might affect results.

**Interoperability.** ModernDiD works with the Python data science ecosystem.
Input data can be pandas or Polars DataFrames. Outputs are NumPy arrays and
named tuples that integrate with standard workflows. Plots use plotnine,
allowing full customization through the grammar of graphics.

Modern causal inference methods should be accessible to everyone doing
empirical research, regardless of their preferred programming language.
ModernDiD aims to make that possible for Python users.
