"""QTrans numerics.arrays helpers."""

import numpy as np


def average_close_values(arr, tol):
    arr = np.asarray(arr)
    if arr.size == 0:
        return arr.copy()
    if np.isnan(arr).any():
        raise ValueError("Input contains NaN, cannot group reliably.")

    sort_idx = np.argsort(arr)
    arr_sorted = arr[sort_idx]

    diffs = np.diff(arr_sorted)
    split_indices = np.where(diffs >= tol)[0] + 1

    arr_grouped = np.empty_like(arr_sorted)

    start = 0
    for end in np.append(split_indices, len(arr_sorted)):
        group = arr_sorted[start:end]

        if group.size > 0:
            avg = group.mean()
            arr_grouped[start:end] = avg

        start = end
    unsort_idx = np.argsort(sort_idx)
    return arr_grouped[unsort_idx]
