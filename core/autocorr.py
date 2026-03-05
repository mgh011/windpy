#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 13:02:11 2026

@author: mauro_ghirardelli
"""

import numpy as np
import xarray as xr
import sys

sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/core/')
from preprocess import get_fluctuations


# ============================================================
# BLOCK A: Public entry point
# ============================================================
def autocorrelation(ds, config, wspd):
    """
    Compute autocorrelation functions and integral length scales.

    The input dataset is expected to be already rotated and gap-filled.
    The function operates on fluctuations (block or detrend), then computes
    autocorrelation of u, v, w over each time window.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing u, v, w (and possibly more), with dims including:
        - time
        - tower
        - height
    config : dict
        Must contain:
        - "window"     : e.g. "10min"
        - "avg_method" : "block" or "detrend" (used in get_fluctuations)
    wspd : xr.DataArray
        Mean wind speed used to convert integral time scale to length scale.
        Typically stats["meanU"], dims (time, tower, height).

    Returns
    -------
    autocorr : xr.Dataset
        Dataset with dims (time, tower, height, lag) and variables:
          - acu, acv, acw
    intlen : xr.Dataset
        Dataset with dims (time, tower, height) and variables:
          - intlenU, intlenV, intlenW
        Computed as lag_at_1_over_e * wspd.
    """
    window = config["window"]

    ds_fluct = get_fluctuations(ds, config)
    autocorr = autocorr_sonic(ds_fluct, window)
    intlen = autocorr_intlen(autocorr, wspd)
    return autocorr, intlen


# ============================================================
# BLOCK B: Autocorrelation per window
# ============================================================
def autocorr_sonic(ds, window, n_lags=200):
    """
    Compute autocorrelation for u, v, w in each resampled time window.

    Returns a Dataset with dims: (time, tower, height, lag).
    """
    for v in ("u", "v", "w"):
        if v not in ds:
            raise ValueError(f"autocorr_sonic requires '{v}' in ds.")

    groups = ds.resample(time=window)

    # get dt and block length from first group
    first_label, first_group = next(iter(groups))
    N = first_group.sizes["time"]
    if N < 4:
        raise ValueError("Time window too short for autocorrelation.")

    dt = (ds.time[1] - ds.time[0]).item() / 1e9

    # logarithmically spaced lags (indices)
    k = np.unique(np.logspace(-1, np.log10(N / 2), num=n_lags).astype(int))
    k = k[(k > 0) & (k < N)]
    lag = k * dt

    out = []
    for label, g in groups:
        g3 = g.transpose("tower", "height", "time")

        u = np.asarray(g3["u"].values)
        v = np.asarray(g3["v"].values)
        w = np.asarray(g3["w"].values)

        # reshape to (n_series, n_time)
        ntow, nh, nt = u.shape
        n_series = ntow * nh

        u2 = u.reshape(n_series, nt)
        v2 = v.reshape(n_series, nt)
        w2 = w.reshape(n_series, nt)

        acu = _calc_autocorr_2d(u2, k).reshape(ntow, nh, -1)
        acv = _calc_autocorr_2d(v2, k).reshape(ntow, nh, -1)
        acw = _calc_autocorr_2d(w2, k).reshape(ntow, nh, -1)

        out.append(
            xr.Dataset(
                coords=dict(
                    time=label,
                    tower=g3["tower"].values,
                    height=g3["height"].values,
                    lag=lag,
                ),
                data_vars=dict(
                    acu=(("tower", "height", "lag"), acu),
                    acv=(("tower", "height", "lag"), acv),
                    acw=(("tower", "height", "lag"), acw),
                ),
            )
        )

    return xr.concat(out, dim="time")


# ============================================================
# BLOCK C: Integral length scale
# ============================================================
def autocorr_intlen(autocorr, wspd):
    """
    Integral length scale based on the first crossing of 1/e.

    intlen = lag_at_min(|acf - exp(-1)|) * wspd
    """
    target = np.exp(-1)

    # find lag where acf is closest to 1/e
    tau = np.abs(autocorr - target).idxmin(dim="lag")

    # tau is in seconds (lag coordinate); multiply by mean speed to get length scale
    intlen = (tau * wspd).rename(dict(acu="intlenU", acv="intlenV", acw="intlenW"))
    return intlen


# ============================================================
# BLOCK D: Helpers
# ============================================================
def _calc_autocorr_2d(x2d, k):
    """
    Autocorrelation for many series at once.

    Parameters
    ----------
    x2d : ndarray, shape (n_series, n_time)
    k : 1D array of lag indices

    Returns
    -------
    ac : ndarray, shape (n_series, n_lags)
    """
    x2d = np.asarray(x2d, dtype=float)
    n_series, N = x2d.shape

    var = np.var(x2d, axis=1)
    var[var == 0] = np.nan  # avoid divide-by-zero

    ac = np.full((n_series, len(k)), np.nan, dtype=float)

    for j, lag in enumerate(k):
        prod = x2d[:, : N - lag] * x2d[:, lag:]
        ac[:, j] = np.nanmean(prod, axis=1) / var

    return ac

