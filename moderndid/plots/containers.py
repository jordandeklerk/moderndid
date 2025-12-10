# pylint: disable=too-many-nested-blocks
"""Data container utilities for PlotCollection."""

import numpy as np


class DataArray:
    """Simplified array container with dimension and coordinate information.

    Parameters
    ----------
    values : ndarray
        The array values.
    dims : list of str
        Names of the dimensions.
    coords : dict of {str: ndarray}, optional
        Coordinate values for each dimension.
    name : str, optional
        Name of this data variable.

    Attributes
    ----------
    values : ndarray
        The stored array values.
    dims : tuple of str
        Dimension names.
    coords : dict
        Coordinate arrays for each dimension.
    name : str or None
        Variable name if set.
    shape : tuple
        Shape of the values array.
    ndim : int
        Number of dimensions.
    """

    def __init__(
        self,
        values,
        dims,
        coords=None,
        name=None,
    ):
        self.values = np.asarray(values)
        self.dims = tuple(dims)
        self.name = name

        if coords is None:
            coords = {}
        self.coords = coords.copy()

        if len(self.dims) != self.values.ndim:
            raise ValueError(f"Number of dims ({len(self.dims)}) must match array ndim ({self.values.ndim})")

        for i, dim in enumerate(self.dims):
            if dim in self.coords:
                if len(self.coords[dim]) != self.values.shape[i]:
                    raise ValueError(
                        f"Coordinate {dim} has length {len(self.coords[dim])} "
                        f"but dimension has size {self.values.shape[i]}"
                    )

    @property
    def shape(self):
        """Shape of the array."""
        return self.values.shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return self.values.ndim

    @property
    def size(self):
        """Total number of elements."""
        return self.values.size

    def sel(self, indexers):
        """Select subset based on coordinate values.

        Parameters
        ----------
        indexers : dict
            Dictionary mapping dimension names to coordinate values or slices.

        Returns
        -------
        DataArray
            New DataArray with selected subset.
        """
        idx = []
        new_dims = []
        new_coords = {}

        for dim in self.dims:
            if dim in indexers:
                indexer = indexers[dim]
                if dim in self.coords:
                    coord = self.coords[dim]
                    if isinstance(indexer, slice):
                        idx.append(indexer)
                        new_dims.append(dim)
                        new_coords[dim] = coord[indexer]
                    else:
                        try:
                            pos = np.where(coord == indexer)[0]
                            if len(pos) == 0:
                                raise KeyError(f"Coordinate value {indexer} not found in {dim}")
                            idx.append(pos[0])
                        except (ValueError, TypeError):
                            idx.append(indexer)
                            if isinstance(indexer, (int, np.integer)):
                                pass
                            else:
                                new_dims.append(dim)
                                new_coords[dim] = coord[indexer]
                else:
                    idx.append(indexer)
                    if not isinstance(indexer, (int, np.integer)):
                        new_dims.append(dim)
            else:
                idx.append(slice(None))
                new_dims.append(dim)
                if dim in self.coords:
                    new_coords[dim] = self.coords[dim]

        new_values = self.values[tuple(idx)]
        return DataArray(new_values, new_dims, new_coords, self.name)

    def item(self):
        """Return the array as a scalar if it has a single element."""
        if self.size != 1:
            raise ValueError(f"Can only convert arrays of size 1 to scalars, got size {self.size}")
        return self.values.item()

    def __repr__(self):
        """Return string representation."""
        return f"DataArray(name={self.name}, dims={self.dims}, shape={self.shape})"


class Dataset:
    """Collection of DataArrays with shared dimensions.

    Parameters
    ----------
    data_vars : dict of {str: DataArray or dict}
        Dictionary mapping variable names to DataArray objects or dicts
        containing 'values', 'dims', and optionally 'coords'.
    coords : dict of {str: ndarray}, optional
        Shared coordinates across variables.

    Attributes
    ----------
    data_vars : dict
        Dictionary of DataArray objects.
    coords : dict
        Shared coordinate arrays.
    dims : set
        Set of all dimension names used across variables.
    """

    def __init__(
        self,
        data_vars,
        coords=None,
    ):
        self.data_vars = {}
        self.coords = coords.copy() if coords is not None else {}

        for name, var in data_vars.items():
            if isinstance(var, dict):
                var_coords = var.get("coords", {})
                merged_coords = self.coords.copy()
                merged_coords.update(var_coords)
                self.data_vars[name] = DataArray(var["values"], var["dims"], merged_coords, name)
            else:
                self.data_vars[name] = var

    @property
    def dims(self):
        """All dimension names across all variables."""
        all_dims = set()
        for da in self.data_vars.values():
            all_dims.update(da.dims)
        return all_dims

    def __getitem__(self, key):
        """Get a data variable by name."""
        return self.data_vars[key]

    def __contains__(self, key):
        """Check if variable exists."""
        return key in self.data_vars

    def sel(self, indexers):
        """Select subset from all variables.

        Parameters
        ----------
        indexers : dict
            Dictionary mapping dimension names to coordinate values.

        Returns
        -------
        Dataset
            New Dataset with all variables subsetted.
        """
        new_vars = {}
        for name, da in self.data_vars.items():
            relevant_indexers = {dim: val for dim, val in indexers.items() if dim in da.dims}
            if relevant_indexers:
                new_vars[name] = da.sel(relevant_indexers)
            else:
                new_vars[name] = da

        return Dataset(new_vars, self.coords)

    def keys(self):
        """Return variable names."""
        return self.data_vars.keys()

    def values(self):
        """Return DataArrays."""
        return self.data_vars.values()

    def items(self):
        """Return (name, DataArray) pairs."""
        return self.data_vars.items()

    def __repr__(self):
        """Return string representation."""
        vars_str = ", ".join(self.data_vars.keys())
        dims_str = ", ".join(sorted(self.dims))
        return f"Dataset(variables=[{vars_str}], dims=[{dims_str}])"


def process_facet_dims(data, facet_dims):
    """Calculate number of facets needed for given dimensions.

    Parameters
    ----------
    data : Dataset
        The dataset to facet.
    facet_dims : list of str
        Dimensions to facet over. Can include '__variable__' to facet by variable.

    Returns
    -------
    n_facets : int
        Total number of facets needed.
    facets_per_var : dict
        If '__variable__' in facet_dims, maps variable names to number of facets
        for that variable. Otherwise empty dict.
    """
    if not facet_dims:
        return 1, {}

    facets_per_var = {}

    if "__variable__" in facet_dims:
        for var_name, da in data.items():
            lengths = [
                len(np.unique(da.coords[dim])) for dim in facet_dims if dim != "__variable__" and dim in da.coords
            ]
            facets_per_var[var_name] = int(np.prod(lengths)) if lengths else 1
        n_facets = sum(facets_per_var.values())
    else:
        missing_dims = {}
        for var_name, da in data.items():
            missing = [dim for dim in facet_dims if dim not in da.dims]
            if missing:
                missing_dims[var_name] = missing

        if missing_dims:
            raise ValueError(f"All variables must have all faceting dimensions, but found missing dims: {missing_dims}")

        lengths = []
        for dim in facet_dims:
            for da in data.values():
                if dim in da.coords:
                    lengths.append(len(np.unique(da.coords[dim])))
                    break
        n_facets = int(np.prod(lengths)) if lengths else 1

    return n_facets, facets_per_var


def iterate_over_selection(data, skip_dims=None):
    """Generate iterator over all coordinate combinations in dataset.

    Parameters
    ----------
    data : Dataset
        Dataset to iterate over.
    skip_dims : set of str, optional
        Dimensions to skip in iteration.

    Returns
    -------
    list of (var_name, sel_dict, isel_dict)
        Each tuple contains variable name, selection dict with coordinate values,
        and isel dict with integer positions.
    """
    if skip_dims is None:
        skip_dims = set()

    results = []

    for var_name, da in data.items():
        iter_dims = [dim for dim in da.dims if dim not in skip_dims]

        if not iter_dims:
            results.append((var_name, {}, {}))
            continue

        coord_arrays = []
        for dim in iter_dims:
            if dim in da.coords:
                coord_arrays.append(da.coords[dim])
            else:
                coord_arrays.append(np.arange(da.shape[da.dims.index(dim)]))

        if len(coord_arrays) == 1:
            indices = [(i,) for i in range(len(coord_arrays[0]))]
        else:
            indices = np.ndindex(*[len(c) for c in coord_arrays])

        for idx in indices:
            if not isinstance(idx, tuple):
                idx = (idx,)

            sel_dict = {}
            isel_dict = {}
            for i, dim in enumerate(iter_dims):
                pos = idx[i]
                isel_dict[dim] = pos
                if dim in da.coords:
                    sel_dict[dim] = da.coords[dim][pos]
                else:
                    sel_dict[dim] = pos

            results.append((var_name, sel_dict, isel_dict))

    return results
