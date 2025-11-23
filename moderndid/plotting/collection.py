# pylint: disable=too-many-nested-blocks
"""PlotCollection class for unified plotting in moderndid."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .aesthetics import generate_aes_mappings, get_aes_kwargs
from .containers import (
    DataArray,
    Dataset,
    iterate_over_selection,
    process_facet_dims,
)


class PlotCollection:
    """Container for managing multi-panel plots with aesthetic mappings.

    Parameters
    ----------
    data : Dataset
        Dataset containing the variables to plot.
    viz : dict
        Dictionary containing visualization elements (figure, plots, artists).
    aes : dict, optional
        Dictionary containing aesthetic mappings.
    backend : str, default="matplotlib"
        Plotting backend. Currently only matplotlib is supported.

    Attributes
    ----------
    data : Dataset
        The data being plotted.
    viz : dict
        Nested dictionary storing figure, plots, and visual elements.
    aes : dict
        Nested dictionary storing aesthetic mappings.
    backend : str
        Plotting backend name.
    coords : dict or None
        Optional coordinate subset for temporary filtering.
    """

    def __init__(
        self,
        data: Dataset,
        viz: dict,
        aes: dict | None = None,
        backend: str = "matplotlib",
    ):
        self.data = data
        self.viz = viz
        self.aes = aes if aes is not None else {}
        self.backend = backend
        self.coords = None

    @property
    def aes_set(self) -> set[str]:
        """Set of all aesthetic keys with mappings defined."""
        return set(self.aes.keys())

    @property
    def facet_dims(self) -> set[str]:
        """Dimensions used for faceting."""
        if "plot" not in self.viz:
            return set()

        plot_data = self.viz["plot"]
        if isinstance(plot_data, dict):
            all_dims = set()
            for var_plots in plot_data.values():
                if isinstance(var_plots, DataArray):
                    all_dims.update(var_plots.dims)
            return all_dims
        if isinstance(plot_data, DataArray):
            return set(plot_data.dims)
        return set()

    @classmethod
    def grid(
        cls,
        data: Dataset,
        cols: list[str] | None = None,
        rows: list[str] | None = None,
        aes: dict[str, list[str]] | None = None,
        figure_kwargs: dict | None = None,
        **aes_kwargs,
    ) -> PlotCollection:
        """Create PlotCollection with grid layout.

        Parameters
        ----------
        data : Dataset
            Dataset to visualize.
        cols : list of str, optional
            Dimensions to use for columns. Can include '__variable__'.
        rows : list of str, optional
            Dimensions to use for rows. Can include '__variable__'.
        aes : dict, optional
            Aesthetic mappings. Maps aesthetic names to dimension lists.
        figure_kwargs : dict, optional
            Keyword arguments passed to matplotlib figure creation.
        **aes_kwargs
            User-provided values for aesthetic properties.

        Returns
        -------
        PlotCollection
            New PlotCollection with grid layout.
        """
        if cols is None:
            cols = []
        if rows is None:
            rows = []
        if aes is None:
            aes = {}
        if figure_kwargs is None:
            figure_kwargs = {}

        if any(dim in rows for dim in cols):
            raise ValueError("Same dimension cannot be in both rows and cols")

        n_cols, _ = process_facet_dims(data, cols)
        n_rows, _ = process_facet_dims(data, rows)

        figsize = figure_kwargs.get("figsize", (4 * n_cols, 3 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        viz = {"figure": fig}

        if "__variable__" not in cols and "__variable__" not in rows:
            dims = tuple(list(rows) + list(cols))
            if dims:
                shape = []
                coords = {}
                for dim in dims:
                    for da in data.values():
                        if dim in da.coords:
                            unique_vals = np.unique(da.coords[dim])
                            shape.append(len(unique_vals))
                            coords[dim] = unique_vals
                            break

                viz["plot"] = DataArray(
                    axes.flatten()[: n_rows * n_cols].reshape(shape) if shape else axes,
                    dims,
                    coords,
                    name="plot",
                )
                viz["row_index"] = DataArray(
                    np.arange(n_rows).repeat(n_cols).reshape(shape) if shape else [[0]],
                    dims,
                    coords,
                    name="row_index",
                )
                viz["col_index"] = DataArray(
                    np.tile(np.arange(n_cols), n_rows).reshape(shape) if shape else [[0]],
                    dims,
                    coords,
                    name="col_index",
                )
            else:
                viz["plot"] = DataArray(axes[0, 0], [], {}, name="plot")
                viz["row_index"] = DataArray(np.array(0), [], {}, name="row_index")
                viz["col_index"] = DataArray(np.array(0), [], {}, name="col_index")
        else:
            viz["plot"] = {}
            viz["row_index"] = {}
            viz["col_index"] = {}

            facet_cumulative = 0
            all_dims = list(rows) + list(cols)

            for var_name, da in data.items():
                var_dims = [d for d in all_dims if d != "__variable__" and d in da.coords]

                if not var_dims:
                    ax = axes.flatten()[facet_cumulative]
                    viz["plot"][var_name] = DataArray(ax, [], {}, name="plot")
                    viz["row_index"][var_name] = DataArray(
                        np.array(facet_cumulative // n_cols), [], {}, name="row_index"
                    )
                    viz["col_index"][var_name] = DataArray(
                        np.array(facet_cumulative % n_cols), [], {}, name="col_index"
                    )
                    facet_cumulative += 1
                else:
                    shape = []
                    coords = {}
                    for dim in var_dims:
                        unique_vals = np.unique(da.coords[dim])
                        shape.append(len(unique_vals))
                        coords[dim] = unique_vals

                    n_facets = int(np.prod(shape)) if shape else 1
                    var_axes = axes.flatten()[facet_cumulative : facet_cumulative + n_facets]

                    viz["plot"][var_name] = DataArray(var_axes.reshape(shape), var_dims, coords, name="plot")

                    indices = np.arange(facet_cumulative, facet_cumulative + n_facets)
                    viz["row_index"][var_name] = DataArray(
                        (indices // n_cols).reshape(shape), var_dims, coords, name="row_index"
                    )
                    viz["col_index"][var_name] = DataArray(
                        (indices % n_cols).reshape(shape), var_dims, coords, name="col_index"
                    )

                    facet_cumulative += n_facets

        aes_dict = generate_aes_mappings(aes, data, **aes_kwargs)

        return cls(data, viz, aes_dict, backend="matplotlib")

    @classmethod
    def wrap(
        cls,
        data: Dataset,
        cols: list[str] | None = None,
        col_wrap: int = 4,
        aes: dict[str, list[str]] | None = None,
        figure_kwargs: dict | None = None,
        **aes_kwargs,
    ) -> PlotCollection:
        """Create PlotCollection with wrapped layout.

        Parameters
        ----------
        data : Dataset
            Dataset to visualize.
        cols : list of str, optional
            Dimensions to iterate over for subplots. Can include '__variable__'.
        col_wrap : int, default=4
            Number of columns before wrapping to new row.
        aes : dict, optional
            Aesthetic mappings.
        figure_kwargs : dict, optional
            Keyword arguments for figure creation.
        **aes_kwargs
            User-provided aesthetic values.

        Returns
        -------
        PlotCollection
            New PlotCollection with wrapped layout.
        """
        if cols is None:
            cols = []
        if aes is None:
            aes = {}
        if figure_kwargs is None:
            figure_kwargs = {}

        n_plots, _ = process_facet_dims(data, cols)

        if n_plots <= col_wrap:
            n_rows, n_cols = 1, n_plots
        else:
            n_rows = (n_plots + col_wrap - 1) // col_wrap
            n_cols = col_wrap

        figsize = figure_kwargs.get("figsize", (4 * n_cols, 3 * n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        flat_axes = axes.flatten()[:n_plots]
        viz = {"figure": fig}

        if "__variable__" not in cols:
            dims = tuple(cols)
            if dims:
                shape = []
                coords = {}
                for dim in dims:
                    for da in data.values():
                        if dim in da.coords:
                            unique_vals = np.unique(da.coords[dim])
                            shape.append(len(unique_vals))
                            coords[dim] = unique_vals
                            break

                viz["plot"] = DataArray(flat_axes.reshape(shape), dims, coords, name="plot")
                viz["row_index"] = DataArray(
                    (np.arange(n_plots) // n_cols).reshape(shape), dims, coords, name="row_index"
                )
                viz["col_index"] = DataArray(
                    (np.arange(n_plots) % n_cols).reshape(shape), dims, coords, name="col_index"
                )
            else:
                viz["plot"] = DataArray(flat_axes[0], [], {}, name="plot")
                viz["row_index"] = DataArray(np.array(0), [], {}, name="row_index")
                viz["col_index"] = DataArray(np.array(0), [], {}, name="col_index")
        else:
            viz["plot"] = {}
            viz["row_index"] = {}
            viz["col_index"] = {}

            facet_cumulative = 0
            for var_name, da in data.items():
                var_dims = [d for d in cols if d != "__variable__" and d in da.coords]

                if not var_dims:
                    ax = flat_axes[facet_cumulative]
                    viz["plot"][var_name] = DataArray(ax, [], {}, name="plot")
                    viz["row_index"][var_name] = DataArray(
                        np.array(facet_cumulative // n_cols), [], {}, name="row_index"
                    )
                    viz["col_index"][var_name] = DataArray(
                        np.array(facet_cumulative % n_cols), [], {}, name="col_index"
                    )
                    facet_cumulative += 1
                else:
                    shape = []
                    coords = {}
                    for dim in var_dims:
                        unique_vals = np.unique(da.coords[dim])
                        shape.append(len(unique_vals))
                        coords[dim] = unique_vals

                    n_facets = int(np.prod(shape)) if shape else 1
                    var_axes = flat_axes[facet_cumulative : facet_cumulative + n_facets]

                    viz["plot"][var_name] = DataArray(var_axes.reshape(shape), var_dims, coords, name="plot")

                    indices = np.arange(facet_cumulative, facet_cumulative + n_facets)
                    viz["row_index"][var_name] = DataArray(
                        (indices // n_cols).reshape(shape), var_dims, coords, name="row_index"
                    )
                    viz["col_index"][var_name] = DataArray(
                        (indices % n_cols).reshape(shape), var_dims, coords, name="col_index"
                    )

                    facet_cumulative += n_facets

        aes_dict = generate_aes_mappings(aes, data, **aes_kwargs)

        return cls(data, viz, aes_dict, backend="matplotlib")

    def map(
        self,
        func: Callable,
        func_label: str | None = None,
        data: str | Dataset | None = None,
        coords: dict | None = None,
        ignore_aes: set[str] | str = frozenset(),
        store_artist: bool = True,
        **kwargs,
    ):
        """Apply a plotting function across all facets and aesthetic combinations.

        Parameters
        ----------
        func : callable
            Function to call for each combination. Should have signature
            func(data, target, **kwargs) where data is the subset data array,
            target is the axes object, and kwargs includes aesthetic mappings.
        func_label : str, optional
            Label for storing the resulting artists. Defaults to func.__name__.
        data : str or Dataset, optional
            Data to plot. If str, looks up variable from self.data.
            If Dataset, uses that. If None, uses self.data.
        coords : dict, optional
            Additional coordinate selection to apply before iteration.
        ignore_aes : set or "all", optional
            Aesthetic mappings to ignore for this function.
        store_artist : bool, default=True
            Whether to store the returned artist objects in viz.
        **kwargs
            Additional keyword arguments passed to func.

        Returns
        -------
        PlotCollection
            Returns self for method chaining.
        """
        if coords is None:
            coords = {}
        if func_label is None:
            func_label = func.__name__
        if isinstance(ignore_aes, str) and ignore_aes == "all":
            ignore_aes = self.aes_set

        if data is None:
            plot_data = self.data
        elif isinstance(data, str):
            if data in self.data:
                da = self.data[data]
                plot_data = Dataset({data: da})
            else:
                raise ValueError(f"Variable '{data}' not found in data")
        else:
            plot_data = data

        aes_to_use = [aes_key for aes_key in self.aes_set if aes_key not in ignore_aes]

        all_loop_dims = set(self.facet_dims)
        for aes_key in aes_to_use:
            aes_data = self.aes[aes_key]
            for aes_val in aes_data.values():
                if isinstance(aes_val, DataArray):
                    all_loop_dims.update(aes_val.dims)

        skip_dims = set(plot_data.dims) - all_loop_dims

        if store_artist:
            self.viz[func_label] = {}

        for var_name, sel, _ in iterate_over_selection(plot_data, skip_dims):
            da = plot_data[var_name]
            if sel:
                da = da.sel(sel)

            target = self._get_target(var_name, {**sel, **coords})
            if target is None:
                continue

            aes_kwargs_dict = get_aes_kwargs(self.aes, aes_to_use, var_name, {**sel, **coords})
            call_kwargs = {**aes_kwargs_dict, **kwargs}
            artist = func(da, target=target, **call_kwargs)

            if store_artist:
                if var_name not in self.viz[func_label]:
                    if sel:
                        var_da = plot_data[var_name]
                        storage_dims = [d for d in var_da.dims if d in all_loop_dims]
                        if storage_dims:
                            storage_shape = [len(np.unique(var_da.coords[d])) for d in storage_dims]
                            storage_coords = {d: np.unique(var_da.coords[d]) for d in storage_dims}
                            self.viz[func_label][var_name] = DataArray(
                                np.full(storage_shape, None, dtype=object),
                                storage_dims,
                                storage_coords,
                                name=func_label,
                            )
                        else:
                            self.viz[func_label][var_name] = DataArray(
                                np.array(None, dtype=object), [], {}, name=func_label
                            )
                    else:
                        self.viz[func_label][var_name] = DataArray(
                            np.array(None, dtype=object), [], {}, name=func_label
                        )

                if isinstance(self.viz[func_label][var_name], DataArray):
                    if sel:
                        storage_da = self.viz[func_label][var_name]
                        idx = []
                        for dim in storage_da.dims:
                            if dim in sel:
                                pos = np.where(storage_da.coords[dim] == sel[dim])[0][0]
                                idx.append(pos)
                            else:
                                idx.append(slice(None))
                        storage_da.values[tuple(idx)] = artist
                    else:
                        self.viz[func_label][var_name].values[()] = artist

        return self

    def _get_target(self, var_name: str | None, selection: dict) -> Any:
        """Get the target axes for a given variable and selection.

        Parameters
        ----------
        var_name : str or None
            Variable name.
        selection : dict
            Coordinate selection.

        Returns
        -------
        matplotlib.axes.Axes or None
            Target axes object.
        """
        if "plot" not in self.viz:
            return None

        plot_data = self.viz["plot"]

        if isinstance(plot_data, dict):
            if var_name not in plot_data:
                return None
            da = plot_data[var_name]
        else:
            da = plot_data

        if isinstance(da, DataArray):
            relevant_sel = {dim: val for dim, val in selection.items() if dim in da.dims}
            if relevant_sel:
                subset = da.sel(relevant_sel)
                return subset.item() if subset.size == 1 else subset.values
            return da.item() if da.size == 1 else da.values
        return da

    def show(self):
        """Display the figure."""
        if "figure" in self.viz:
            plt.show()

    def savefig(self, filename: str, **kwargs):
        """Save the figure to file.

        Parameters
        ----------
        filename : str
            Output filename.
        **kwargs
            Passed to matplotlib's savefig.
        """
        if "figure" in self.viz:
            fig = self.viz["figure"]
            fig.savefig(filename, **kwargs)
