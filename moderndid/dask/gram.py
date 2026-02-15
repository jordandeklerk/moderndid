"""Gram-matrix based influence function representation for analytical SE computation.

For boot=False, all aggregation SE computations reduce to quadratic forms using
only the tiny Gram matrix V = IF^T @ IF instead of the full IF matrix.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from moderndid.dask.ddd import _build_sparse_inf, _searchsorted_sparse


class GramIF:
    """Gram-matrix representation of the influence function matrix."""

    def __init__(self, V, col_sums, M_g, group_counts, glist_orig, n, valid_col_indices=None):
        self.V = np.asarray(V, dtype=np.float64)
        self.col_sums = np.asarray(col_sums, dtype=np.float64)
        self.M_g = np.asarray(M_g, dtype=np.float64)
        self.group_counts = np.asarray(group_counts, dtype=np.float64)
        self.glist_orig = np.asarray(glist_orig)
        self._n = int(n)
        self._n_cols = self.V.shape[0]
        self._valid_col_indices = np.asarray(valid_col_indices) if valid_col_indices is not None else None

    @property
    def shape(self):
        """Return (n_units, n_cols) shape tuple."""
        return (self._n, self._n_cols)

    def copy(self):
        """Return self (immutable representation)."""
        return self

    def __getitem__(self, key):
        """Subset by columns: ``gram[:, col_indices]``."""
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            if isinstance(row_key, slice) and row_key == slice(None):
                col_indices = np.atleast_1d(np.asarray(col_key))
                return GramIF(
                    V=self.V[np.ix_(col_indices, col_indices)],
                    col_sums=self.col_sums[col_indices],
                    M_g=self.M_g[col_indices, :],
                    group_counts=self.group_counts,
                    glist_orig=self.glist_orig,
                    n=self._n,
                )
        raise IndexError(f"GramIF only supports [:, col_indices] slicing, got {key}")


def compute_gram_from_sparse_cols(sparse_cols, n_units, n_columns, unit_groups, glist, valid_col_indices=None):
    """Build a GramIF from per-column sparse data."""
    CSC = _build_sparse_inf(sparse_cols, n_units, n_columns)
    if valid_col_indices is not None:
        CSC = CSC[:, valid_col_indices]

    V = (CSC.T @ CSC).toarray()
    col_sums = np.array(CSC.sum(axis=0)).ravel()

    n_groups = len(glist)
    group_counts = np.array([np.sum(unit_groups == g) for g in glist])

    rows = []
    cols = []
    for j, g in enumerate(glist):
        mask = np.where(unit_groups == g)[0]
        rows.append(mask)
        cols.append(np.full(len(mask), j))
    if rows:
        G_rows = np.concatenate(rows)
        G_cols = np.concatenate(cols)
        G = sp.csc_matrix((np.ones(len(G_rows)), (G_rows, G_cols)), shape=(n_units, n_groups))
    else:
        G = sp.csc_matrix((n_units, n_groups))

    M_g = (CSC.T @ G).toarray()

    return GramIF(V, col_sums, M_g, group_counts, glist, n_units)


def gram_se(V, keepers, weights, n):
    """sqrt(w^T V[keepers,keepers] w / n^2)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    V_sub = V[np.ix_(keepers, keepers)]
    return float(np.sqrt(w @ V_sub @ w / (n * n)))


def gram_cross_term(V, wh_a, w_a, wh_b, w_b):
    """w_a^T V[wh_a, wh_b] w_b."""
    w_a = np.asarray(w_a, dtype=np.float64).ravel()
    w_b = np.asarray(w_b, dtype=np.float64).ravel()
    return float(w_a @ V[np.ix_(wh_a, wh_b)] @ w_b)


def gram_overall_se_from_parts(V, keepers_list, weights_list, overall_weights, n):
    """Overall SE for multi-element aggregation (eventstudy, calendar).

    Var = sum_jk ow_j ow_k * w_j^T V[k_j, k_k] w_k;  SE = sqrt(Var / n^2).
    """
    n_elements = len(keepers_list)
    var = 0.0
    for j in range(n_elements):
        for k in range(n_elements):
            cross = gram_cross_term(
                V,
                keepers_list[j],
                weights_list[j],
                keepers_list[k],
                weights_list[k],
            )
            var += overall_weights[j] * overall_weights[k] * cross
    return float(np.sqrt(var / (n * n)))


def gram_group_overall_se(
    gram_if, keepers_per_group, weights_per_group, att_g_clean, pgg, orig_glist, unit_groups_glist_keepers, n
):
    """Overall SE for group aggregation: Var = A^T A + 2 A^T B + B^T B.

    A = IF_g_mat @ w_overall,  B = wif @ att_g_clean.
    """
    V = gram_if.V
    col_sums = gram_if.col_sums
    M_g = gram_if.M_g
    group_counts = gram_if.group_counts
    glist_orig = gram_if.glist_orig

    weights_overall = pgg / pgg.sum()
    n_groups = len(pgg)

    ata = 0.0
    for i in range(n_groups):
        ki = keepers_per_group[i]
        wi = weights_per_group[i]
        if ki is None:
            continue
        for j in range(n_groups):
            kj = keepers_per_group[j]
            wj = weights_per_group[j]
            if kj is None:
                continue
            ata += weights_overall[i] * weights_overall[j] * gram_cross_term(V, ki, wi, kj, wj)

    wTw = _gram_wif_T_wif(unit_groups_glist_keepers, pgg, group_counts, glist_orig, orig_glist, n)
    btb = att_g_clean @ wTw @ att_g_clean

    IF_T_wif = _gram_IF_T_wif(M_g, col_sums, unit_groups_glist_keepers, pgg, glist_orig, orig_glist)
    atb = 0.0
    for i in range(n_groups):
        ki = keepers_per_group[i]
        wi = weights_per_group[i]
        if ki is None:
            continue
        row = wi @ IF_T_wif[np.ix_(ki, np.arange(n_groups))]
        atb += weights_overall[i] * (row @ att_g_clean)

    var = max(ata + 2 * atb + btb, 0.0)
    return float(np.sqrt(var / (n * n)))


def compute_gram_distributed(client, worker_data_futures, n_units, spec_to_col, valid_col_indices, unit_groups, glist):
    """Compute GramIF from worker-side IF Futures (Level 2)."""
    n_valid = len(valid_col_indices)
    col_to_valid = {int(c): i for i, c in enumerate(valid_col_indices)}

    partial_futures = []
    for spec_idx, (data_fut, worker) in worker_data_futures.items():
        if data_fut is None:
            continue
        valid_j = col_to_valid.get(spec_to_col[spec_idx])
        if valid_j is None:
            continue
        fut = client.submit(
            _compute_partial_gram,
            data_fut,
            valid_j,
            n_valid,
            glist,
            unit_groups,
            workers=[worker],
            allow_other_workers=False,
            pure=False,
        )
        partial_futures.append(fut)

    V = np.zeros((n_valid, n_valid))
    col_sums = np.zeros(n_valid)
    n_groups = len(glist)
    M_g = np.zeros((n_valid, n_groups))
    group_counts = np.array([np.sum(unit_groups == g) for g in glist])

    for partial in client.gather(partial_futures):
        if partial is not None:
            p_V, p_col_sums, p_M_g = partial
            V += p_V
            col_sums += p_col_sums
            M_g += p_M_g

    return GramIF(V, col_sums, M_g, group_counts, glist, n_units)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _gram_IF_T_wif(M_g, col_sums, keepers_agg, pg, glist_gram, orig_glist):
    """Compute IF^T @ wif analytically from M_g and col_sums.

    IF^T @ if1[:, j] = (M_g[:, kj] - pg[j] * col_sums) / sum_pg
    IF^T @ if2[:, j] = (M_g_keepers_sum - sum_pg * col_sums) * pg[j] / sum_pg^2
    """
    sum_pg = pg[keepers_agg].sum()
    n_cols = len(col_sums)
    n_agg = len(keepers_agg)

    gram_col_map = {float(g): j for j, g in enumerate(glist_gram)}
    keeper_gram_indices = [gram_col_map.get(float(orig_glist[ki]), -1) for ki in keepers_agg]

    M_g_keepers_sum = np.zeros(n_cols)
    for idx in keeper_gram_indices:
        if idx >= 0:
            M_g_keepers_sum += M_g[:, idx]

    result = np.zeros((n_cols, n_agg))
    for j_agg, ki in enumerate(keepers_agg):
        gram_idx = keeper_gram_indices[j_agg]
        pg_j = pg[ki]

        if1_col = (M_g[:, gram_idx] - pg_j * col_sums) / sum_pg if gram_idx >= 0 else -pg_j * col_sums / sum_pg

        if2_col = (M_g_keepers_sum - sum_pg * col_sums) * pg_j / (sum_pg**2)
        result[:, j_agg] = if1_col - if2_col

    return result


def _gram_wif_T_wif(keepers_agg, pg, group_counts, glist_gram, orig_glist, n):
    """Compute wif^T @ wif analytically from group statistics.

    Decomposes into if1^T if1 - if1^T if2 - if2^T if1 + if2^T if2.
    """
    sum_pg = pg[keepers_agg].sum()
    n_agg = len(keepers_agg)

    gram_count_map = {float(g): c for g, c in zip(glist_gram, group_counts, strict=True)}
    keeper_counts = np.array(
        [gram_count_map.get(float(orig_glist[ki]), 0) for ki in keepers_agg],
        dtype=np.float64,
    )
    pg_k = pg[keepers_agg]

    if1_T_if1 = np.diag(keeper_counts).astype(np.float64)
    if1_T_if1 -= np.outer(pg_k, keeper_counts)
    if1_T_if1 -= np.outer(keeper_counts, pg_k)
    if1_T_if1 += n * np.outer(pg_k, pg_k)
    if1_T_if1 /= sum_pg**2

    n_in_keepers = keeper_counts.sum()
    rs_sq = n_in_keepers * (1 - sum_pg) ** 2 + (n - n_in_keepers) * sum_pg**2

    if2_T_if2 = np.outer(pg_k, pg_k) * rs_sq / (sum_pg**4)

    sum_row_sums = n_in_keepers - n * sum_pg
    ind_T_rs = keeper_counts * (1 - sum_pg)

    if1_T_if2 = np.zeros((n_agg, n_agg))
    for j in range(n_agg):
        cross_j = ind_T_rs[j] - pg_k[j] * sum_row_sums
        if1_T_if2[j, :] = pg_k * cross_j / (sum_pg**3)

    return if1_T_if1 - if1_T_if2 - if1_T_if2.T + if2_T_if2


def _split_result_on_worker(cell_result, sorted_ids):
    """Split a cell result into (driver_payload, worker_data) on the worker.

    driver_payload = (att_entry, diag_v, se_val) -- tiny, transferred to driver.
    worker_data = (row_idx, vals) -- stays on worker for distributed V.
    """
    if cell_result is None:
        return (None, None, None), None

    att_entry, inf_data, se_val = cell_result
    if att_entry is None:
        return (None, None, None), None

    if inf_data is not None:
        inf_func_scaled, cell_id_list = inf_data
        row_idx, vals = _searchsorted_sparse(inf_func_scaled, cell_id_list, sorted_ids)
        diag_v = float(np.dot(vals, vals))
        return (att_entry, diag_v, se_val), (row_idx, vals)

    return (att_entry, None, se_val), None


def _compute_partial_gram(sparse_data, col_j, n_valid, glist, unit_groups):
    """Worker-side: partial V, col_sums, M_g from a single column."""
    if sparse_data is None:
        return None

    row_idx, vals = sparse_data

    V_partial = np.zeros((n_valid, n_valid))
    col_sums_partial = np.zeros(n_valid)
    M_g_partial = np.zeros((n_valid, len(glist)))

    col_sums_partial[col_j] = vals.sum()
    V_partial[col_j, col_j] = np.dot(vals, vals)

    for g_idx, g in enumerate(glist):
        group_mask = unit_groups[row_idx] == g
        if group_mask.any():
            M_g_partial[col_j, g_idx] = vals[group_mask].sum()

    return V_partial, col_sums_partial, M_g_partial
