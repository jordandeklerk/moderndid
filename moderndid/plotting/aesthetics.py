"""Aesthetic mapping system for PlotCollection."""

from itertools import cycle

import numpy as np

from moderndid.plotting.containers import DataArray

DEFAULT_AES = {
    "color": ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"],
    "marker": ["o", "s", "^", "v", "D", "P", "X", "*", "p", "h"],
    "linestyle": ["-", "--", "-.", ":"],
    "alpha": [1.0, 0.8, 0.6, 0.4],
    "linewidth": [1.5, 2.0, 2.5, 3.0],
    "markersize": [6, 8, 10, 12],
}


def get_default_aes_values(aes_key, n_values, kwargs):
    """Get default aesthetic values for a given aesthetic.

    Parameters
    ----------
    aes_key : str
        Name of the aesthetic (e.g., 'color', 'marker').
    n_values : int
        Number of unique values needed.
    kwargs : dict
        User-provided aesthetic values. If aes_key is in kwargs,
        those values are used instead of defaults.

    Returns
    -------
    list
        List of aesthetic values of length n_values.
    """
    if aes_key in kwargs:
        user_values = kwargs[aes_key]
        if not isinstance(user_values, (list, tuple, np.ndarray)):
            user_values = [user_values]
    else:
        user_values = DEFAULT_AES.get(aes_key, [None])

    values = []
    cycler = cycle(user_values)
    for _ in range(n_values):
        values.append(next(cycler))

    return values


def generate_aes_mappings(
    aes,
    data,
    **kwargs,
):
    """Generate aesthetic mappings from dimension specifications.

    Parameters
    ----------
    aes : dict
        Dictionary mapping aesthetic names to lists of dimensions.
        Use '__variable__' as a pseudo-dimension to map over variables.
        Use False as value to disable an aesthetic.
    data : Dataset
        Dataset containing the data to map aesthetics for.
    **kwargs
        User-provided aesthetic value cycles.

    Returns
    -------
    dict
        Nested dictionary structure with aesthetic mappings.
        Structure: {aes_key: {var_name or 'mapping': DataArray, ...}}
    """
    aes_dict = {}

    aes = {key: value for key, value in aes.items() if value is not False}

    all_dims = set()
    for dims in aes.values():
        if isinstance(dims, (list, tuple)):
            all_dims.update(d for d in dims if d != "__variable__")

    coords = {}
    for dim in all_dims:
        for da in data.values():
            if dim in da.coords:
                unique_vals = np.unique(da.coords[dim])
                if dim not in coords:
                    coords[dim] = unique_vals
                break

    for aes_key, dims in aes.items():
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        if "__variable__" in dims:
            aes_dict[aes_key] = {}

            total_aes_vals = 0
            for var_name, da in data.items():
                aes_dims = [d for d in dims if d != "__variable__" and d in da.coords]
                if aes_dims:
                    shape = [len(np.unique(da.coords[d])) for d in aes_dims]
                    total_aes_vals += int(np.prod(shape))
                else:
                    total_aes_vals += 1

            aes_vals = get_default_aes_values(aes_key, total_aes_vals, kwargs)

            aes_cumulative = 0
            for var_name, da in data.items():
                aes_dims = [d for d in dims if d != "__variable__" and d in da.coords]

                if not aes_dims:
                    aes_dict[aes_key][var_name] = DataArray(np.array(aes_vals[aes_cumulative]), [], {}, name=aes_key)
                    aes_cumulative += 1
                else:
                    shape = [len(np.unique(da.coords[d])) for d in aes_dims]
                    n_vals = int(np.prod(shape))
                    values = np.array(aes_vals[aes_cumulative : aes_cumulative + n_vals]).reshape(shape)
                    aes_coords = {d: np.unique(da.coords[d]) for d in aes_dims}
                    aes_dict[aes_key][var_name] = DataArray(values, aes_dims, aes_coords, name=aes_key)
                    aes_cumulative += n_vals

        else:
            aes_dict[aes_key] = {}

            aes_dims_in_var = {
                var_name: all(d in da.dims or d in da.coords for d in dims) for var_name, da in data.items()
            }

            if not any(aes_dims_in_var.values()):
                shape = [len(coords[d]) for d in dims]
                n_vals = int(np.prod(shape)) if shape else 1
                aes_vals = get_default_aes_values(aes_key, n_vals + 1, kwargs)

                aes_dict[aes_key]["neutral_element"] = DataArray(np.array(aes_vals[0]), [], {}, name=aes_key)
                aes_dict[aes_key]["mapping"] = DataArray(
                    np.array(aes_vals[1 : n_vals + 1]).reshape([len(coords[d]) for d in dims]),
                    dims,
                    {d: coords[d] for d in dims},
                    name=aes_key,
                )
            elif all(aes_dims_in_var.values()):
                shape = [len(coords[d]) for d in dims]
                n_vals = int(np.prod(shape)) if shape else 1
                aes_vals = get_default_aes_values(aes_key, n_vals, kwargs)

                aes_dict[aes_key]["mapping"] = DataArray(
                    np.array(aes_vals).reshape([len(coords[d]) for d in dims]),
                    dims,
                    {d: coords[d] for d in dims},
                    name=aes_key,
                )
            else:
                shape = [len(coords[d]) for d in dims]
                n_vals = int(np.prod(shape)) if shape else 1
                aes_vals = get_default_aes_values(aes_key, n_vals + 1, kwargs)

                aes_dict[aes_key]["neutral_element"] = DataArray(np.array(aes_vals[0]), [], {}, name=aes_key)

                mapping_vals = [v for v in aes_vals[1:] if v != aes_vals[0]]
                if not mapping_vals:
                    mapping_vals = aes_vals[1 : n_vals + 1]
                else:
                    while len(mapping_vals) < n_vals:
                        mapping_vals.append(mapping_vals[len(mapping_vals) % len(mapping_vals)])
                    mapping_vals = mapping_vals[:n_vals]

                aes_dict[aes_key]["mapping"] = DataArray(
                    np.array(mapping_vals).reshape([len(coords[d]) for d in dims]),
                    dims,
                    {d: coords[d] for d in dims},
                    name=aes_key,
                )

    return aes_dict


def get_aes_kwargs(
    aes_dict,
    aes_keys,
    var_name,
    selection,
):
    """Extract aesthetic values for a specific variable and selection.

    Parameters
    ----------
    aes_dict : dict
        Aesthetic mappings from generate_aes_mappings.
    aes_keys : list of str
        Which aesthetics to retrieve values for.
    var_name : str
        Name of the variable.
    selection : dict
        Dictionary mapping dimension names to coordinate values.

    Returns
    -------
    dict
        Dictionary mapping aesthetic names to values for this combination.
    """
    kwargs = {}

    for aes_key in aes_keys:
        if aes_key.startswith("overlay"):
            continue

        if aes_key not in aes_dict:
            continue

        aes_data = aes_dict[aes_key]

        if var_name in aes_data:
            da = aes_data[var_name]
            relevant_sel = {dim: val for dim, val in selection.items() if dim in da.dims}
            if relevant_sel:
                subset = da.sel(relevant_sel)
                kwargs[aes_key] = subset.item() if subset.size == 1 else subset.values
            else:
                kwargs[aes_key] = da.item() if da.size == 1 else da.values

        elif "mapping" in aes_data:
            mapping_da = aes_data["mapping"]
            relevant_sel = {dim: val for dim, val in selection.items() if dim in mapping_da.dims}

            if all(dim in relevant_sel for dim in mapping_da.dims):
                subset = mapping_da.sel(relevant_sel)
                kwargs[aes_key] = subset.item() if subset.size == 1 else subset.values
            elif "neutral_element" in aes_data:
                kwargs[aes_key] = aes_data["neutral_element"].values.item()
            else:
                raise ValueError(
                    f"Cannot get aesthetic {aes_key} for {var_name}: "
                    f"dimensions {mapping_da.dims} not in selection {selection.keys()} "
                    "and no neutral element available"
                )
        elif "neutral_element" in aes_data:
            kwargs[aes_key] = aes_data["neutral_element"].values.item()

    return kwargs
