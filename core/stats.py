#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:08:26 2026

@author: mauro_ghirardelli
"""

# windy/core/stats.py

import xarray as xr
import sys

#import local libraries
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/core/')
from preprocess import get_fluctuations


def fluxes_calc(ds, config):
    """
    Compute Block-1 statistics for pressure–velocity analysis.

    Supported/expected fields
    -------------------------
    Required:
      - u, v, w
    Expected (present in your datasets):
      - tc
    Optional:
      - P

    Output (per window block)
    -------------------------
    Means:
      - meanU : mean(u)
      - meanT : mean(tc)    (if tc exists)
      - meanP : mean(P)     (if P exists)

    Second moments / covariances (from fluctuations):
      - uu, vv, ww, uv, uw, vw
      - TT, uT, vT, wT       (if tc exists)
      - pp, pu, pv, pw, pT   (if P exists; pT only if tc exists)

    Derived:
      - tke   : 0.5*(uu+vv+ww)
      - ustar : (uw^2 + vw^2)^(1/4)

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with dimension 'time' (and possibly height,tower).
    config : dict
        Must contain "window". Should contain "avg_method".

    Returns
    -------
    xr.Dataset
        Statistics on the resampled time grid (one value per window).
    """
    window = config["window"]

    for v in ("u", "v", "w"):
        if v not in ds.data_vars:
            raise ValueError(f"fluxes_calc requires variable '{v}' in dataset.")

    has_t = "tc" in ds.data_vars
    has_p = "P" in ds.data_vars

    # fluctuations
    ds_fluct = get_fluctuations(ds, config)

    # instantaneous products to be averaged over window
    out = {
        "meanU": ds["u"],
        "uu": ds_fluct["u"] * ds_fluct["u"],
        "vv": ds_fluct["v"] * ds_fluct["v"],
        "ww": ds_fluct["w"] * ds_fluct["w"],
        "uv": ds_fluct["u"] * ds_fluct["v"],
        "uw": ds_fluct["u"] * ds_fluct["w"],
        "vw": ds_fluct["v"] * ds_fluct["w"],
        "tke": 0.5 * (ds_fluct["u"] ** 2 + ds_fluct["v"] ** 2 + ds_fluct["w"] ** 2),
    }

    if has_t:
        out.update(
            {
                "meanT": ds["tc"],
                "TT": ds_fluct["tc"] * ds_fluct["tc"],
                "uT": ds_fluct["u"] * ds_fluct["tc"],
                "vT": ds_fluct["v"] * ds_fluct["tc"],
                "wT": ds_fluct["w"] * ds_fluct["tc"],
            }
        )

    if has_p:
        out.update(
            {
                "meanP": ds["P"],
                "pp": ds_fluct["P"] * ds_fluct["P"],
                "pu": ds_fluct["P"] * ds_fluct["u"],
                "pv": ds_fluct["P"] * ds_fluct["v"],
                "pw": ds_fluct["P"] * ds_fluct["w"],
            }
        )
        if has_t:
            out["pT"] = ds_fluct["P"] * ds_fluct["tc"]

    flux = xr.Dataset(data_vars=out)

    # block average
    flux = flux.resample(time=window).mean()

    # derived friction velocity (needs uw and vw)
    flux = flux.assign(ustar=(flux["uw"] ** 2 + flux["vw"] ** 2) ** 0.25)
    
    # --- Monin–Obukhov length and stability ---
    kappa = 0.4
    g = 9.81
    eps = 1e-12
    
    # mean temperature in Kelvin (heuristic: if looks like Celsius, convert)
    Tmean = flux["meanT"]
    Tmean_K = xr.where(Tmean < 100.0, Tmean + 273.15, Tmean)
    
    # wT must exist to compute L
    if "wT" in flux.data_vars:
        L = - (flux["ustar"] ** 3) * Tmean_K / (kappa * g * (flux["wT"] + eps))
        flux = flux.assign(L=L)


    return flux
