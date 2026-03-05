#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 10:06:16 2026

@author: mauro_ghirardelli
"""

# windy/core/spectra.py

import numpy as np
import xarray as xr
from scipy import signal
import sys

sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/core/')
from preprocess import get_fluctuations


def _infer_dt_seconds(da_time: xr.DataArray):
    dt = da_time.diff("time")
    dt_s = (dt / np.timedelta64(1, "s")).median(skipna=True)
    return float(dt_s.values)



def _resolve_welch_params(n, segments=None, overlap=0.0):
    if not segments:
        return n, 0
    ov = float(overlap)
    ov = max(0.0, min(0.95, ov))
    denom = (1.0 + (segments - 1) * (1.0 - ov))
    nperseg = max(8, int(np.floor(n / denom)))
    noverlap = int(np.floor(ov * nperseg))
    return nperseg, noverlap


def _welch_2d(x2d, fs, *, segments=None, overlap=0.0, window="hann", detrend="constant", scaling="density"):
    """
    x2d: shape (n_series, n_time)
    returns freq (nf,), psd (n_series, nf) without f=0
    """
    n = x2d.shape[-1]
    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    f, Pxx = signal.welch(
        x2d, fs=fs, axis=-1,
        window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap,
        scaling=scaling
    )
    return f[1:], np.real(Pxx[..., 1:])


def _csd_2d(x2d, y2d, fs, *, segments=None, overlap=0.0, window="hann", detrend="constant", scaling="density"):
    """
    x2d,y2d: shape (n_series, n_time)
    returns freq (nf,), Cxy (n_series, nf) without f=0
    """
    n = x2d.shape[-1]
    nperseg, noverlap = _resolve_welch_params(n, segments=segments, overlap=overlap)
    f, Pxy = signal.csd(
        x2d, y2d, fs=fs, axis=-1,
        window=window, detrend=detrend,
        nperseg=nperseg, noverlap=noverlap,
        scaling=scaling
    )
    return f[1:], Pxy[..., 1:]


def spectra_welch(ds_fluct, window, *, segments=3, overlap=0.5, window_type="hann", detrend="constant"):
    """
    Welch autospectra + selected cospectra, block-by-block.

    Returns xr.Dataset with dims: time, tower, height, freq
    (order may be time,tower,height,freq).
    """
    # normalize time dim name (we assume caller already renamed time_10hz -> time)
    if "time" not in ds_fluct.dims:
        raise ValueError("spectra_welch expects dimension 'time'.")

    # required vars
    for v in ("u", "v", "w", "tc"):
        if v not in ds_fluct.data_vars:
            raise ValueError(f"spectra_welch requires '{v}' in ds_fluct.")
    has_p = "P" in ds_fluct.data_vars

    dt = _infer_dt_seconds(ds_fluct["time"])
    fs = 1.0 / dt

    groups = ds_fluct.resample(time=window)
    out_blocks = []

    for label, g in groups:
        # bring to (tower, height, time) for fast reshape
        # if a dim is missing (unlikely), this will error loudly (good)
        g3 = g.transpose("tower", "height", "time")

        u = np.asarray(g3["u"].values)
        v = np.asarray(g3["v"].values)
        w = np.asarray(g3["w"].values)
        tc = np.asarray(g3["tc"].values)
        P = np.asarray(g3["P"].values) if has_p else None

        # reshape to (n_series, n_time)
        nt = u.shape[-1]
        n_series = u.shape[0] * u.shape[1]
        u2 = u.reshape(n_series, nt)
        v2 = v.reshape(n_series, nt)
        w2 = w.reshape(n_series, nt)
        t2 = tc.reshape(n_series, nt)
        p2 = P.reshape(n_series, nt) if has_p else None

        # autospectra
        freq, su = _welch_2d(u2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sv = _welch_2d(v2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sw = _welch_2d(w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, sT = _welch_2d(t2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

        # cospectra (main) - complex cross spectra
        _, cuw = _csd_2d(u2, w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cvw = _csd_2d(v2, w2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cuv = _csd_2d(u2, v2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
        _, cwT = _csd_2d(w2, t2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

        data = {
            # autospectra
            "su": su, "sv": sv, "sw": sw, "sT": sT,

            # keep REAL cospectra as before (backward compatible)
            "cuw": cuw.real, "cvw": cvw.real, "cuv": cuv.real, "cwT": cwT.real,

            # NEW: imaginary (quadrature) parts
            "cuw_im": cuw.imag, "cvw_im": cvw.imag, "cuv_im": cuv.imag, "cwT_im": cwT.imag,
        }

        if has_p:
            _, sp = _welch_2d(p2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)
            _, cwp = _csd_2d(w2, p2, fs, segments=segments, overlap=overlap, window=window_type, detrend=detrend)

            data["sp"] = sp

            # keep REAL cospectrum as before
            data["cwp"] = cwp.real

            # NEW: imaginary (quadrature) part
            data["cwp_im"] = cwp.imag


        # back to (tower,height,freq)
        ntow = g3.sizes["tower"]
        nh = g3.sizes["height"]
        nf = len(freq)

        ds_block = xr.Dataset(
            coords=dict(
                time=label,
                tower=g3["tower"].values,
                height=g3["height"].values,
                freq=freq,
            )
        )

        for name, arr in data.items():
            ds_block[name] = (("tower", "height", "freq"), arr.reshape(ntow, nh, nf))

        out_blocks.append(ds_block)

    return xr.concat(out_blocks, dim="time")


def spectral_slopes_epsilon(spectra, Umean):
    """
    Compute epsilon + slopes from autospectra su,sv,sw,sT using a cutoff f_c = U/(2π z).

    spectra: Dataset with dims (time,tower,height,freq) and vars su,sv,sw,sT
    Umean: DataArray with dims (time,tower,height) (or broadcastable)
    """
    if "height" not in spectra.coords:
        raise ValueError("spectral_slopes_epsilon expects coord 'height'.")
    z = spectra["height"]

    # cutoff broadcast: (time,tower,height)
    cutoff = Umean / (2.0 * np.pi * z)

    S = spectra[["su", "sv", "sw", "sT"]]
    S = S.where(S > 0)   # elimina zeri e negativi


    # high freq range: f > cutoff AND avoid last bins
    fmax = spectra["freq"].isel(freq=-8)
    Sh = S.where((spectra.freq > cutoff) & (spectra.freq < fmax))

    # push left limit to first maximum (per component)
    # (keeps the original idea, but simplified: use su peak)
    f_peak = Sh["su"].idxmax(dim="freq")
    Sh = Sh.where(spectra.freq > f_peak)

    Sl = S.where(spectra.freq < cutoff)

    # constants
    cu = 18 / 55 * 1.5
    cvw = cu * 4 / 3
    cT = 0.8

    # epsilon (median over freq)
    epsU = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.su / cu)**(3/2)).median("freq").rename("epsU")
    epsV = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.sv / cvw)**(3/2)).median("freq").rename("epsV")
    epsW = (2*np.pi/Umean * (Sh.freq**(5/3) * Sh.sw / cvw)**(3/2)).median("freq").rename("epsW")
    epsT = (((2*np.pi/Umean)**(2/3)) * (Sh.freq**(5/3)) * Sh.sT * (epsU**(1/3)) / cT).median("freq").rename("epsT")

    epsilon = xr.merge([epsU, epsV, epsW, epsT])

    # slopes: fit log10(S) vs log10(f)
    Sh_log = np.log10(Sh).assign_coords(freq=np.log10(Sh.freq))
    Sl_log = np.log10(Sl).assign_coords(freq=np.log10(Sl.freq))

    slopes_h = Sh_log.polyfit("freq", deg=1).sel(degree=1).drop_vars("degree").rename(
        dict(
            su_polyfit_coefficients="slopeHU",
            sv_polyfit_coefficients="slopeHV",
            sw_polyfit_coefficients="slopeHW",
            sT_polyfit_coefficients="slopeHT",
        )
    )

    slopes_l = Sl_log.polyfit("freq", deg=1).sel(degree=1).drop_vars("degree").rename(
        dict(
            su_polyfit_coefficients="slopeLU",
            sv_polyfit_coefficients="slopeLV",
            sw_polyfit_coefficients="slopeLW",
            sT_polyfit_coefficients="slopeLT",
        )
    )

    slopes = xr.merge([slopes_h, slopes_l])
    return slopes, epsilon


def spectra_eps(ds, config, Umean, *, welch_segments=3, welch_overlap=0.5):
    """
    Entry point: computes Welch spectra + epsilon/slopes.
    """
    window = config["window"]

    # Optional: override defaults from config["spectra"]["welch"]
    spec_cfg = config.get("spectra", {})
    welch_cfg = spec_cfg.get("welch", {})

    welch_segments = int(welch_cfg.get("segments", welch_segments))
    welch_overlap = float(welch_cfg.get("overlap", welch_overlap))
    window_type = welch_cfg.get("window", "hann")
    
    # IMPORTANT: ds_fluct is already mean-removed or detrended per window.
    # Do NOT detrend again inside Welch (would change low-f energy inconsistently).
    detrend = False

    ds_fluct = get_fluctuations(ds, config)

    welch = spectra_welch(
        ds_fluct,
        window,
        segments=welch_segments,
        overlap=welch_overlap,
        window_type=window_type,
        detrend=detrend,
    )

    slopes, epsilon = spectral_slopes_epsilon(welch, Umean)
    return welch, epsilon, slopes



def bin_spectra_log(ds, N_bin=80, freq_name="freq", out_freq_name="freq_bin", mode="area"):
    """
    Log-bin all spectral variables along frequency.

    mode:
      - "area": area-preserving PSD binning: S_bin = (1/Δf) ∫ S(f) df
      - "shape": geometric mean in log-space (preserves shape/slope for positive spectra)
    """

    if freq_name not in ds.coords:
        raise ValueError(f"bin_spectra_log requires coordinate '{freq_name}'.")

    freq0 = np.asarray(ds[freq_name].values)
    if freq0.ndim != 1:
        raise ValueError(f"Expected 1D coord '{freq_name}', got shape {freq0.shape}")

    # edges from valid positive freqs
    valid_f = np.isfinite(freq0) & (freq0 > 0)
    fpos = freq0[valid_f]
    if fpos.size < 2:
        raise ValueError("Not enough positive finite frequencies to bin.")

    edges = np.logspace(np.log10(fpos.min()), np.log10(fpos.max()), N_bin + 1)
    freq_bin = np.sqrt(edges[:-1] * edges[1:])  # geometric bin centers

    def _bin_1d(spec_1d):
        spec_1d = np.asarray(spec_1d)

        # IMPORTANT: freq and spectrum must match length
        if spec_1d.shape[0] != freq0.shape[0]:
            raise ValueError(
                f"bin_spectra_log: spectrum length {spec_1d.shape[0]} != freq length {freq0.shape[0]}"
            )

        if mode == "shape":
            valid = np.isfinite(freq0) & (freq0 > 0) & np.isfinite(spec_1d) & (spec_1d > 0)
        else:
            valid = np.isfinite(freq0) & (freq0 > 0) & np.isfinite(spec_1d)

        f = freq0[valid]
        s = spec_1d[valid]

        out = np.full(N_bin, np.nan, dtype=float)
        if f.size < 2:
            return out

        idx = np.digitize(f, edges) - 1
        ok = (idx >= 0) & (idx < N_bin)
        f = f[ok]; s = s[ok]; idx = idx[ok]

        for b in range(N_bin):
            m = (idx == b)
            if not np.any(m):
                continue

            fb = f[m]
            sb = s[m]

            if mode == "shape":
                out[b] = float(np.exp(np.nanmean(np.log(sb))))
            else:
                # area-preserving PSD
                order = np.argsort(fb)
                fb = fb[order]; sb = sb[order]
                df = fb[-1] - fb[0]
                if df > 0:
                    out[b] = float(np.trapz(sb, x=fb) / df)

        return out

    out_vars = {}
    for name, da in ds.data_vars.items():
        if freq_name not in da.dims:
            out_vars[name] = da
            continue

        binned = xr.apply_ufunc(
            _bin_1d,
            da,
            input_core_dims=[[freq_name]],
            output_core_dims=[[out_freq_name]],
            vectorize=True,
            dask="allowed",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {out_freq_name: N_bin}},
        ).assign_coords({out_freq_name: freq_bin})

        out_vars[name] = binned

    out = xr.Dataset(data_vars=out_vars, coords=dict(ds.coords))
    out = out.drop_vars(freq_name, errors="ignore")
    out = out.assign_coords({out_freq_name: freq_bin})

    # keep canonical name 'freq'
    out = out.rename({out_freq_name: freq_name})

    return out


