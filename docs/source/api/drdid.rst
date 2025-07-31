.. _drdid:

Doubly Robust DiD
=================

.. note::
   Top level functions are designed to be accessible by the user. The low level API functions which power the top level functions are intended for advanced users who want to customize their estimators beyond the provided wrappers and are displayed in the API reference for clarity.

The ``drdid`` module provides doubly robust estimators for the average treatment effect on the treated (ATT) in difference-in-differences (DID) designs, based on the work of `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_. These estimators are consistent if either a propensity score model or an outcome regression model is correctly specified, but not necessarily both, offering robustness against model misspecification.

Background
----------

Setup and Notation
~~~~~~~~~~~~~~~~~~

We consider a setting with two groups and two time periods. Let :math:`Y_{it}` be the outcome for unit :math:`i` at time :math:`t`, where :math:`t=0` is the pre-treatment period and :math:`t=1` is the post-treatment period. Let :math:`D_i` be an indicator for treatment status, where :math:`D_i=1` if the unit is in the treatment group and :math:`D_i=0` for the comparison group. We assume treatment happens between :math:`t=0` and :math:`t=1`, so :math:`D_{i0}=0` for all :math:`i`. We also observe a vector of pre-treatment covariates :math:`X_i`.

Using the potential outcomes framework, :math:`Y_{it}(d)` is the outcome that would be observed for unit :math:`i` at time :math:`t` under treatment status :math:`d`. The observed outcome is :math:`Y_{it} = D_i Y_{it}(1) + (1-D_i) Y_{it}(0)`.

The parameter of interest is the Average Treatment Effect on the Treated (ATT)

.. math::
   \tau = \mathbb{E}[Y_{1}(1) - Y_{1}(0) | D=1].

Since :math:`Y_1(1)` is observed for the treated group, we can write

.. math::
   \tau = \mathbb{E}[Y_1 | D=1] - \mathbb{E}[Y_1(0) | D=1].

The main identification challenge is to estimate the counterfactual term :math:`\mathbb{E}[Y_1(0) | D=1]`.

Identification
~~~~~~~~~~~~~~

The key identifying assumptions are:

1.  **Data Structure**: The data are assumed to be either

    (a) A panel dataset where :math:`\{Y_{i0}, Y_{i1}, D_i, X_i\}_{i=1}^n` are independent and identically distributed (i.i.d.).

    (b) A pooled repeated cross-section where :math:`\{Y_i, D_i, X_i, T_i\}_{i=1}^n` are i.i.d. draws from a mixture distribution, and the joint distribution of :math:`(D, X)` is invariant to the time period :math:`T`.

2.  **Conditional Parallel Trends Assumption (PTA)**: The average outcomes for the treated and comparison groups would have evolved in parallel, conditional on covariates

    .. math::
       \mathbb{E}[Y_1(0) - Y_0(0) | D=1, X] = \mathbb{E}[Y_1(0) - Y_0(0) | D=0, X].

3.  **Overlap**: For all values of covariates :math:`X`, there is a positive probability of being in either the treatment or comparison group

    .. math::
       \mathbb{P}(D=1|X) < 1-\varepsilon \text{ for some } \varepsilon > 0.

Doubly Robust Estimands
~~~~~~~~~~~~~~~~~~~~~~~

The doubly robust estimators combine the strengths of the outcome regression (OR) and inverse probability weighting (IPW) approaches. The resulting estimator for the ATT is consistent if either the outcome regression model or the propensity score model is correctly specified, but not necessarily both.

Let :math:`p(X) = \mathbb{P}(D=1|X)` be the propensity score and :math:`\pi(X)` be a working model for the propensity score.

**Panel Data**

When panel data are available, we observe :math:`(Y_{i0}, Y_{i1})` for each unit. Let :math:`\Delta Y = Y_1 - Y_0` and :math:`\mu_{0, \Delta}^p(X)` be a working model for the outcome evolution of the comparison group, :math:`\mathbb{E}[\Delta Y | D=0, X]`.

The DR estimand for panel data is given by

.. math::
   \tau^{dr, p} = \mathbb{E}\left[ (w_1^p(D) - w_0^p(D, X; \pi)) (\Delta Y - \mu_{0, \Delta}^p(X)) \right],

where the weights are defined as

.. math::
   w_1^p(D) = \frac{D}{\mathbb{E}[D]} \quad \text{and} \quad w_0^p(D, X; \pi) = \frac{\frac{\pi(X)(1-D)}{1-\pi(X)}}{\mathbb{E}\left[\frac{\pi(X)(1-D)}{1-\pi(X)}\right]}.

This estimand is consistent for the ATT if either the propensity score model is correct, :math:`\pi(X) = p(X)`, or the outcome model is correct, :math:`\mu_{0, \Delta}^p(X) = \mathbb{E}[\Delta Y | D=0, X]`.

**Repeated Cross-Sections**

When only repeated cross-sections are available, we do not observe the same units in both periods. Let :math:`T` be a time indicator with :math:`T=1` for post-treatment and :math:`T=0` for pre-treatment. Let :math:`\mu_{d,t}^{rc}(X)` be a working model for :math:`\mathbb{E}[Y | D=d, T=t, X]`, and define

.. math::
   \mu_{d, Y}^{rc}(T, X) = T \cdot \mu_{d, 1}^{rc}(X) + (1-T) \cdot \mu_{d, 0}^{rc}(X).

Two DR estimands are proposed. The first, :math:`\tau_1^{dr,rc}`, is given by

.. math::
   \tau_{1}^{dr, rc} = \mathbb{E}\left[ (w_{1}^{rc}(D, T) - w_{0}^{rc}(D, T, X; \pi)) (Y - \mu_{0, Y}^{rc}(T, X)) \right].

The second estimand, :math:`\tau_2^{dr,rc}`, which is locally efficient, is

.. math::
   \begin{aligned}
   \tau_{2}^{dr, rc} = \tau_{1}^{dr, rc} &+ \left(\mathbb{E}[\mu_{1,1}^{rc}(X)-\mu_{0,1}^{rc}(X) | D=1] - \mathbb{E}[\mu_{1,1}^{rc}(X)-\mu_{0,1}^{rc}(X) | D=1, T=1]\right) \\
   &- \left(\mathbb{E}[\mu_{1,0}^{rc}(X)-\mu_{0,0}^{rc}(X) | D=1] - \mathbb{E}[\mu_{1,0}^{rc}(X)-\mu_{0,0}^{rc}(X) | D=1, T=0]\right)
   \end{aligned}

The weights for the repeated cross-sections case are defined as

.. math::
   w_{1, t}^{rc}(D, T) = \frac{D \cdot \mathbf{1}\{T=t\}}{\mathbb{E}[D \cdot \mathbf{1}\{T=t\}]} \quad \text{and} \quad w_{0, t}^{rc}(D, T, X; \pi) = \frac{\frac{\pi(X)(1-D) \cdot \mathbf{1}\{T=t\}}{1-\pi(X)}}{\mathbb{E}\left[\frac{\pi(X)(1-D) \cdot \mathbf{1}\{T=t\}}{1-\pi(X)}\right]}.

Both estimands are consistent for the ATT under the same doubly robust conditions: if either the propensity score model or the outcome model for the comparison group is correctly specified.

.. note::
   For the full theoretical details, including efficiency bounds and asymptotic properties, please refer to the original paper by `Sant'Anna and Zhao (2020) <https://psantanna.com/files/SantAnna_Zhao_DRDID.pdf>`_.

Top Level Functions
-------------------

.. currentmodule:: didpy

.. autosummary::
   :toctree: generated/
   :recursive:

   drdid
   ipwdid
   ordid

Panel Data Estimators
---------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   drdid_imp_panel
   drdid_panel
   ipw_did_panel
   reg_did_panel
   std_ipw_did_panel
   twfe_did_panel

Repeated Cross-Sections Estimators
----------------------------------

.. autosummary::
   :toctree: generated/
   :recursive:

   drdid_imp_local_rc
   drdid_imp_rc
   drdid_rc
   drdid_trad_rc
   ipw_did_rc
   reg_did_rc
   std_ipw_did_rc
   twfe_did_rc
