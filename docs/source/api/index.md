# API Reference

Documentation for all public modules, classes, and functions in the pyDiD package.

## Doubly Robust DiD Estimators

```{eval-rst}
.. autosummary::
   :toctree: generated/

   pydid.drdid_imp_panel
```

## Core Propensity Estimators

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

   pydid.ipw_did_rc
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
