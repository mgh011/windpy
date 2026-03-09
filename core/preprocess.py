#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:07:43 2026

@author: mauro_ghirardelli
"""

# windy/core/preprocess.py
import numpy as np
import xarray as xr
from scipy import signal


# MAIN
def get_fluctuations(ds, config):
    """
    Construct turbulent fluctuations x'(t) from raw time series.
    
    This function defines the fluctuation signal x'(t) using a single, explicit
    choice controlled by:
    
        config["fluctuations"]["mode"]
    
    The mode is mutually exclusive (only one definition of fluctuations is applied).
    
    ------------------------------------------------------------------------
    FLUCTUATION MODES (mutually exclusive)
    ------------------------------------------------------------------------
    Valid values of config["fluctuations"]["mode"] are:
    
      • "block"
            x'(t) = x(t) − ⟨x⟩_window
            Removes the mean within each averaging window (e.g. 30 min).
            This is the classical Reynolds decomposition using a window mean.
    
      • "detrend"
            x'(t) = x(t) − (a + b t)
            Removes a linear trend independently within each averaging window.
            More aggressive than "block" in removing low-frequency variability.
    
      • "butterworth"
            x'(t) = HP[x(t)]
            Applies a zero-phase Butterworth high-pass filter directly to the raw
            signal, using the cutoff and order specified in
            config["fluctuations"]["butterworth"].
            No additional block-mean removal or detrending is performed.
    
      • "raw"
            x'(t) = x(t)
            No mean removal. Intended for diagnostics or debugging only.
    
    ------------------------------------------------------------------------
    WINDOW PARAMETER
    ------------------------------------------------------------------------
    For modes "block" and "detrend", fluctuations are defined independently within
    time windows specified by:
    
        config["fluctuations"]["window"]   (e.g. "30min")
    
    For "butterworth" and "raw", the window parameter is not used to define x'(t),
    but may still be required elsewhere in the processing pipeline (e.g. resampling
    of statistics, stationarity tests, or spectral aggregation).
    
    ------------------------------------------------------------------------
    SPECTRAL CONSISTENCY
    ------------------------------------------------------------------------
    The output of this function MUST be used consistently for both:
    
        - bulk statistics (variance, fluxes, moments)
        - spectral estimation (Welch)
    
    Welch should therefore be called with detrend=False, because mean/trend removal
    (if any) is already performed here. This supports approximate consistency:
    
        var[x']  ≈  ∫ S_x'(f) df
    
    up to windowing and finite-band effects.
    
    ------------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------------
    ds : xarray.Dataset
        Raw time series dataset (time as a dimension).
    
    config : dict
        Configuration dictionary containing the "fluctuations" section:
    
            config["fluctuations"] = {
                "mode": "block" | "detrend" | "butterworth" | "raw",
                "window": "30min",                      # required for block/detrend
                "butterworth": {                        # required for butterworth
                    "cutoff_seconds": 300.0,
                    "order": 4
                }
            }
    
    ------------------------------------------------------------------------
    Returns
    ------------------------------------------------------------------------
    ds_fluct : xarray.Dataset
        Dataset containing the fluctuation signals x'(t) for all variables.
    
    ------------------------------------------------------------------------
    Notes
    ------------------------------------------------------------------------
    • The choice of mode changes the physical definition of variance and strongly
      affects the low-frequency part of spectra.
    • "detrend" removes more low-frequency energy than "block".
    • "butterworth" explicitly defines a frequency cutoff and should be treated
      as a different physical bandpass definition of turbulence.
    """
    
    ds0 = ds.copy()

    # ------------------------------------------------------------
    # NEW unified fluctuations config
    # ------------------------------------------------------------
    fluc_cfg = config.get("fluctuations", None)

    if fluc_cfg is None:
        raise ValueError(
            "Config must contain 'fluctuations' section."
        )

    mode = fluc_cfg.get("mode", "block").lower() #if mode exists, use mode. Else use "block" (default)
    window = config["window"]

    # ------------------------------------------------------------
    # BUTTERWORTH (HP only)
    # ------------------------------------------------------------
    if mode == "butterworth":

        bw_cfg = fluc_cfg.get("butterworth", {})
        cutoff_seconds = float(bw_cfg.get("cutoff_seconds", 300.0))
        order = int(bw_cfg.get("order", 4))
    
        return butterworth_highpass(
            ds0,
            cutoff_seconds=cutoff_seconds,
            order=order
        )

    # ------------------------------------------------------------
    # BLOCK MEAN
    # ------------------------------------------------------------
    elif mode in ("block", "mean"):

        ds_mean = ds0.resample(time=window).mean()
        ds_mean = ds_mean.reindex(time=ds0.time).ffill(dim="time")
        return ds0 - ds_mean

    # ------------------------------------------------------------
    # DETREND
    # ------------------------------------------------------------
    elif mode == "detrend":

        return detrend(ds0, window)

    # ------------------------------------------------------------
    # RAW (no mean removal)
    # ------------------------------------------------------------
    elif mode in ("raw", "none"):

        return ds0

    else:
        raise ValueError(
            f"Unknown fluctuations.mode = {mode}"
        )

#Helpers 

def detrend(ds, window):
    """
    Linear detrend within each resampled time block along the *time* dimension,
    regardless of dimension order.
    """
    groups = ds.resample(time=window)
    out_blocks = []

    for _, g in groups:
        # detrend each variable along the 'time' axis explicitly
        g_dt = xr.Dataset()
        for v in g.data_vars:
            da = g[v]
            ax = da.get_axis_num("time")  # <-- the key fix
            g_dt[v] = xr.apply_ufunc(
                signal.detrend,
                da,
                kwargs={"axis": ax, "type": "linear"},
                dask="allowed",
                output_dtypes=[da.dtype],
            )
        out_blocks.append(g_dt)

    return xr.concat(out_blocks, dim="time")


def butterworth_highpass(ds, cutoff_seconds, order=4):
    """
    Apply zero-phase Butterworth high-pass filter to all data variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimension 'time'.
    cutoff_seconds : float
        Cutoff period in seconds (e.g. 300 for 5 min).
    order : int
        Butterworth filter order.

    Returns
    -------
    xarray.Dataset
        High-pass filtered dataset (time domain).
    """
    
    # infer sampling frequency
    dt = (ds.time.diff("time") / np.timedelta64(1, "s")).median().item()
    fs = 1.0 / dt

    fc = 1.0 / cutoff_seconds
    wn = fc / (0.5 * fs)  # normalized cutoff

    b, a = signal.butter(order, wn, btype="highpass")

    out = xr.Dataset(coords=ds.coords)

    for v in ds.data_vars:
        da = ds[v]
        axis = da.get_axis_num("time")

        # ---- REMOVE MEAN FIRST (important for stability) ----
        da = da - da.mean(dim="time")

        filtered = xr.apply_ufunc(
            signal.filtfilt,
            b, a, da,
            kwargs={"axis": axis},
            dask="allowed",
            output_dtypes=[da.dtype],
        )

        out[v] = filtered

    return out