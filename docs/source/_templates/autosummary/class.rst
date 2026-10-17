{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :exclude-members: count, index
{%- if '__call__' in all_methods %}
   :special-members: __call__
{%- endif %}

{# Inherited members such as the accessors that BSpline gets from SplineBase stay on the page.
Since count and index come from the tuple behind each NamedTuple result, the page leaves them out. #}
