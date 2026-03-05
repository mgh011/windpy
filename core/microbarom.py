#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:21:37 2026

@author: mauro_ghirardelli
"""


import numpy as np
import xarray as xr


def _mb_area_1d(freq, sp, *, f_min, f_max, fit_deg, fit_range):
    """
    1D microbarom area diagnostic on one spectrum.

    Fit log10(sp) vs log10(f) outside [f_min, f_max] within fit_range,
    then integrate fit and residuals inside the microbarom band.
    """
    freq = np.asarray(freq)
    sp = np.asarray(sp)

    valid = np.isfinite(freq) & np.isfinite(sp) & (freq > 0) & (sp > 0)
    f = freq[valid]
    y = sp[valid]
    if f.size < fit_deg + 2:
        return np.nan, np.nan, np.nan

    fit_mask = (
        (f >= fit_range[0]) & (f <= fit_range[1]) &
        ((f < f_min) | (f > f_max))
    )
    f_fit = f[fit_mask]
    y_fit = y[fit_mask]
    if f_fit.size < fit_deg + 2:
        return np.nan, np.nan, np.nan

    # fit in log-log
    coeffs = np.polyfit(np.log10(f_fit), np.log10(y_fit), deg=fit_deg)

    # evaluate fit over eval range
    eval_mask = (f >= fit_range[0]) & (f <= fit_range[1])
    f_eval = f[eval_mask]
    y_eval = y[eval_mask]
    y_fit_all = 10 ** np.polyval(coeffs, np.log10(f_eval))

    # band
    band = (f_eval >= f_min) & (f_eval <= f_max)
    f_band = f_eval[band]
    y_band = y_eval[band]
    fit_band = y_fit_all[band]
    if f_band.size < 2:
        return np.nan, np.nan, np.nan

    area_fit = np.trapz(fit_band, x=f_band)
    area_peak = np.trapz(np.maximum(y_band - fit_band, 0.0), x=f_band)
    area_peak_abs = np.trapz(np.abs(y_band - fit_band), x=f_band)
    return area_fit, area_peak, area_peak_abs


def get_microbarom(
    spectra,
    stats=None,
    *,
    f_min=0.1,
    f_max=0.6,
    fit_deg=3,
    fit_range=(0.01, 5.0),
):
    """
    Compute microbarom diagnostics from pressure autospectra.

    Parameters
    ----------
    spectra : xr.Dataset
        Spectral dataset with dims (time, tower, height, freq) and variable 'sp'
        (pressure PSD, Pa^2/Hz).
    stats : xr.Dataset, optional
        Optional stats dataset (not used here, kept for interface compatibility).
    f_min, f_max : float
        Microbarom band bounds [Hz].
    fit_deg : int
        Polynomial degree for background fit in log-log space.
    fit_range : (float, float)
        Frequency range used for fit and evaluation [Hz].

    Returns
    -------
    xr.Dataset
        Dataset with dims (time, tower, height) and variables:
          - area_fit
          - area_peak
          - area_peak_abs
    """
    if "sp" not in spectra.data_vars:
        raise ValueError("get_microbarom requires 'sp' in spectra (pressure autospectrum).")
    if "freq" not in spectra.coords:
        raise ValueError("get_microbarom requires coordinate 'freq'.")

    sp = spectra["sp"]

    needed = {"time", "tower", "height", "freq"}
    if not needed.issubset(set(sp.dims)):
        raise ValueError(f"'sp' must have dims including {needed}, got {sp.dims}")

    sp4 = sp.transpose("time", "tower", "height", "freq")
    freq = spectra["freq"].values

    nt, ntow, nh, nf = sp4.shape
    area_fit = np.full((nt, ntow, nh), np.nan, dtype=float)
    area_peak = np.full_like(area_fit, np.nan)
    area_peak_abs = np.full_like(area_fit, np.nan)

    sp_vals = np.asarray(sp4.values)

    for ti in range(nt):
        for toi in range(ntow):
            for hi in range(nh):
                af, ap, apa = _mb_area_1d(
                    freq, sp_vals[ti, toi, hi, :],
                    f_min=f_min, f_max=f_max,
                    fit_deg=fit_deg, fit_range=fit_range
                )
                area_fit[ti, toi, hi] = af
                area_peak[ti, toi, hi] = ap
                area_peak_abs[ti, toi, hi] = apa

    return xr.Dataset(
        data_vars=dict(
            area_fit=(("time", "tower", "height"), area_fit),
            area_peak=(("time", "tower", "height"), area_peak),
            area_peak_abs=(("time", "tower", "height"), area_peak_abs),
        ),
        coords=dict(
            time=sp4["time"],
            tower=sp4["tower"],
            height=sp4["height"],
        ),
    )

