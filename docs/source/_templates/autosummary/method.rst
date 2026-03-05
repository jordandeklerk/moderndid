:orphan:

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

method

.. auto{{ objtype }}:: {{ fullname | replace("moderndid.", "moderndid::") }}

{# In the fullname (e.g. `moderndid.didcont.spline.BSpline.basis`), the module
name is ambiguous. Using a `::` separator specifies `moderndid` as the module
name. #}
