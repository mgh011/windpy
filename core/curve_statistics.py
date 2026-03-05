#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:26:46 2026

@author: mauro_ghirardelli
"""

import numpy as np


def median_curve(
    x_list,
    y_list,
    n_grid=80,
    min_frac=0.2,
):
    """
    Compute a 'median' curve using median in log(y).

    Parameters
    ----------
    x_list : list of 1D arrays
        List of x arrays (one per curve).
    y_list : list of 1D arrays
        List of y arrays (same length as x_list).
        y must be strictly positive where valid.
    n_grid : int
        Number of log-spaced points in k grid.
    min_frac : float
        Minimum fraction of curves required to compute median (e.g. 0.2 = 20%).

    Returns
    -------
    x_grid : 1D array
        Log-spaced x grid (min–max over this bin).
    y_med : 1D array
        Median curve (in original y units). NaN where coverage insufficient.
    """

    # --- number of curves
    N = len(x_list)
    if N == 0:
        raise ValueError("No curves provided.")

    # --- collect global k min/max
    x_mins = []
    x_maxs = []

    for x in x_list:
        if len(x) == 0:
            continue
        x_valid = x[np.isfinite(x) & (x > 0)]
        if len(x_valid) == 0:
            continue
        x_mins.append(x_valid.min())
        x_maxs.append(x_valid.max())

    if len(x_mins) == 0:
        raise ValueError("No valid x values found.")

    x_min = min(x_mins)
    x_max = max(x_maxs)

    if x_min <= 0 or x_max <= x_min:
        raise ValueError("Invalid x range.")

    # --- fixed grid (log-spaced)
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), n_grid)

    # --- storage for interpolated curves
    Y = np.full((N, n_grid), np.nan)

    for j, (x, y) in enumerate(zip(x_list, y_list)):

        x = np.asarray(x)
        y = np.asarray(y)

        mask = (
            np.isfinite(x) & np.isfinite(y) &
            (x > 0) & (y > 0)
        )

        if mask.sum() < 2:
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        # sort (important for interpolation)
        idx = np.argsort(x_valid)
        x_valid = x_valid[idx]
        y_valid = y_valid[idx]

        # interpolation only inside curve range
        x_lo = x_valid.min()
        x_hi = x_valid.max()

        inside = (x_grid >= x_lo) & (x_grid <= x_hi)

        if inside.sum() == 0:
            continue

        Y[j, inside] = np.interp(
            x_grid[inside],
            x_valid,
            y_valid
        )

    # --- compute median in log-space
    y_med = np.full(n_grid, np.nan)
    y_p25 = np.full(n_grid, np.nan)
    y_p75 = np.full(n_grid, np.nan)

    min_required = int(np.ceil(min_frac * N))

    for i in range(n_grid):
        vals = Y[:, i]
        vals = vals[np.isfinite(vals) & (vals > 0)]

        if len(vals) >= min_required:
            logs = np.log10(vals)

            y_med[i] = 10 ** np.median(logs)
            y_p25[i] = 10 ** np.percentile(logs, 25)
            y_p75[i] = 10 ** np.percentile(logs, 75)
        # else remains NaN

    return x_grid, y_med, y_p25, y_p75