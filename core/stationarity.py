#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:08:33 2026

@author: mauro_ghirardelli
"""

# windy/core/stationarity.py

import numpy as np
import xarray as xr
import sys

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/core/')
from preprocess import get_fluctuations



def covariance_for_stationarity(ds):
    """
    Covariances (block-level) used by stationarity test.

    Returns an xr.Dataset with variables:
      - sig_u : std(u) computed from covariance definition
      - uw
      - vw
      - wT  (if tc exists)
      - pw  (if P exists)
    """
    u = ds["u"]
    v = ds["v"]
    w = ds["w"]

    sig_u = np.sqrt((u * u).mean("time") - u.mean("time") * u.mean("time"))
    uw = (u * w).mean("time") - u.mean("time") * w.mean("time")
    vw = (v * w).mean("time") - v.mean("time") * w.mean("time")

    out = dict(sig_u=sig_u, uw=uw, vw=vw)

    if "tc" in ds.data_vars:
        tc = ds["tc"]
        out["wT"] = (w * tc).mean("time") - w.mean("time") * tc.mean("time")

    if "P" in ds.data_vars:
        P = ds["P"]
        out["pw"] = (P * w).mean("time") - P.mean("time") * w.mean("time")

    return xr.Dataset(data_vars=out)


def _subwindow_sixth(window):
    """
    Sub-window = (window / 6) expressed as seconds string, robust for '10min', '1h', etc.
    """
    t = xr.cftime_range("2000-01-01", periods=2, freq=window)
    dt = (t[1] - t[0]).total_seconds()
    sub_seconds = dt / 6.0
    return f"{int(round(sub_seconds))}S"


def stationarity(ds, config):
    """
    Stationarity test computed on sub-intervals (1/6 of the main window).

    For each window block:
      stat = 100 * abs( (cov_block - mean(cov_subblocks)) / cov_block )

    Returns
    -------
    xr.Dataset on the resampled time grid, with:
      - statSigU, statUW, statVW, (optional statWT, statPW)
    """
    window = config["window"]

    sub_window = _subwindow_sixth(window)

    ds_fluct = get_fluctuations(ds, config)
    groups = ds_fluct.resample(time=window)

    out = []

    for label, g in groups:
        cov_block = covariance_for_stationarity(g).assign_coords(time=label)

        sub_groups = g.resample(time=sub_window)
        sub_covs = [covariance_for_stationarity(sg) for _, sg in sub_groups]

        cov_sub_mean = xr.concat(sub_covs, dim="time").mean("time").assign_coords(time=label)

        stat = 100 * np.abs((cov_block - cov_sub_mean) / cov_block)
        out.append(stat)

    stat_ds = xr.concat(out, dim="time")

    rename_map = {"sig_u": "statSigU", "uw": "statUW", "vw": "statVW"}
    if "wT" in stat_ds.data_vars:
        rename_map["wT"] = "statWT"
    if "pw" in stat_ds.data_vars:
        rename_map["pw"] = "statPW"

    return stat_ds.rename(rename_map)
