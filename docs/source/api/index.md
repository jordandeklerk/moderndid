# API Reference

Documentation for all public modules, classes, and functions in the pyDiD package.

## Doubly-Robust DiD Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.drdid
   pydid.drdid_imp_local_rc
   pydid.drdid_imp_rc
   pydid.drdid_rc
   pydid.drdid_trad_rc
   pydid.drdid_imp_panel
   pydid.drdid_panel
```

## Inverse-Propensity Weighted DiD Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.ipwdid
   pydid.ipw_did_panel
   pydid.ipw_did_rc
   pydid.std_ipw_did_rc
   pydid.std_ipw_did_panel
```

## Outcome Regression DiD Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.ordid
   pydid.reg_did_panel
   pydid.reg_did_rc
```

## Propensity Estimators

### IPT Propensity Estimator

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.calculate_pscore_ipt
```

### AIPW Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.aipw_did_panel
   pydid.aipw_did_rc_imp1
   pydid.aipw_did_rc_imp2
```

### IPW Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.ipw_rc
```

## Bootstrap Estimators

### Panel Data Bootstrap

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.wboot_drdid_imp_panel
   pydid.wboot_dr_tr_panel
   pydid.wboot_ipw_panel
   pydid.wboot_std_ipw_panel
   pydid.wboot_reg_panel
   pydid.wboot_twfe_panel
```

### Repeated Cross-Section Bootstrap

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.wboot_drdid_rc1
   pydid.wboot_drdid_rc2
   pydid.wboot_drdid_ipt_rc1
   pydid.wboot_drdid_ipt_rc2
   pydid.wboot_ipw_rc
   pydid.wboot_std_ipw_rc
   pydid.wboot_reg_rc
   pydid.wboot_twfe_rc
```

### Standard Multiplier Bootstrap

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.mboot_did
   pydid.mboot_twfep_did
```

## Supporting Functions

### Weighted OLS

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.wols_panel
   pydid.wols_rc
```
