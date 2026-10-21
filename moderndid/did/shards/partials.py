"""Combination rules for the statistics that shards contribute to a pass."""

import numpy as np

from .container import Partial


def empty_partial():
    """Return a partial that holds no statistics."""
    return Partial(sums={}, maxima={}, factors={}, keyed={})


def combine_partials(first, second):
    """Combine the statistics of two disjoint sets of units.

    A statistic that only one set holds passes through unchanged. Two factor
    pairs :math:`(R_1, Q_1'B_1)` and :math:`(R_2, Q_2'B_2)` stack into one
    matrix whose QR factorization gives the pair for all the rows, as in a
    tall-skinny QR factorization.

    Parameters
    ----------
    first : Partial
        Statistics of one set of units.
    second : Partial
        Statistics of another set of units.

    Returns
    -------
    Partial
        Statistics of the units of both sets:

        - **sums**: Sums of the two sets added up
        - **maxima**: Elementwise maximum of the two sets' maxima
        - **factors**: Factor pair of all the rows of both sets
        - **keyed**: Keyed sums with the rows that share a key added up
    """
    sums = dict(first.sums)
    for name, value in second.sums.items():
        sums[name] = sums[name] + value if name in sums else value

    maxima = dict(first.maxima)
    for name, value in second.maxima.items():
        maxima[name] = np.maximum(maxima[name], value) if name in maxima else value

    factors = dict(first.factors)
    for name, value in second.factors.items():
        factors[name] = _stack_factors(factors[name], value) if name in factors else value

    keyed = dict(first.keyed)
    for name, value in second.keyed.items():
        keyed[name] = _merge_keyed(keyed[name], value) if name in keyed else value

    return Partial(sums=sums, maxima=maxima, factors=factors, keyed=keyed)


def finalize_keyed(partial):
    r"""Replace each keyed sum by the Gram matrix of its rows and its number of keys.

    A keyed sum named ``name`` with rows :math:`s_c` becomes the sum
    :math:`\sum_c s_c s_c'` under ``name_gram`` and the number of keys under
    ``name_count``. Since a key's row is final only once every unit with that
    key has arrived, a set of shards can finalize its own keyed sums only when
    it holds every unit of its keys. The Gram matrices of disjoint sets of keys
    then add up.

    Parameters
    ----------
    partial : Partial
        Statistics whose keyed sums hold every unit of their keys.

    Returns
    -------
    Partial
        The same statistics with the keyed sums moved into the sums:

        - **sums**: The sums of ``partial`` with ``name_gram`` and ``name_count`` added for each keyed sum ``name``
        - **maxima**: The maxima of ``partial``
        - **factors**: The factors of ``partial``
        - **keyed**: An empty dict
    """
    sums = dict(partial.sums)
    for name, (keys, rows) in partial.keyed.items():
        for suffix, value in (("gram", rows.T @ rows), ("count", np.array(float(len(keys))))):
            entry = f"{name}_{suffix}"
            sums[entry] = sums[entry] + value if entry in sums else value
    return Partial(sums=sums, maxima=partial.maxima, factors=partial.factors, keyed={})


def _stack_factors(first, second):
    """Refactor two stacked triangular factors so that one pair describes all their rows."""
    triangular = np.vstack([first[0], second[0]])
    transformed = np.vstack([first[1], second[1]])
    # Since a set of units with no rows of a design contributes no factor, the other pair already describes them all.
    if first[0].shape[0] == 0 or second[0].shape[0] == 0:
        return triangular, transformed
    q, r = np.linalg.qr(triangular)
    return r, q.T @ transformed


def _merge_keyed(first, second):
    """Add up the rows of two keyed sums that share a key."""
    keys = np.concatenate([first[0], second[0]])
    rows = np.vstack([first[1], second[1]])
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    merged = np.column_stack(
        [np.bincount(inverse, weights=rows[:, j], minlength=len(unique_keys)) for j in range(rows.shape[1])]
    )
    return unique_keys, merged.reshape(len(unique_keys), rows.shape[1])
