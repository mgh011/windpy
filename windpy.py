#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import xarray as xr
import sys

# local path (come fai tu)
sys.path.append('/Users/mauro_ghirardelli/Documents/windpy/core/')

from layout import check_dataset_structure
from gapfill import fill_gaps
from rotation import double_rotation
from stats import fluxes_calc
from stationarity import stationarity
from spectra import spectra_eps, bin_spectra_log
from microbarom import get_microbarom
from anisotropy import anisotropy_from_stats
from autocorr import autocorrelation


# ============================================================
# CONFIG
# ============================================================

def load_config_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# LOAD ONE HF FILE (QUESTO È QUELLO CHE FUNZIONA)
# ============================================================

def load_single_hf_file(cfg, filename):
    hf_dir = cfg["paths"]["hf_dir"]
    full_path = os.path.join(hf_dir, filename)

    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    wanted_vars = tuple(cfg.get("fields", ["P", "u", "v", "w", "tc"]))

    ds = xr.open_dataset(
        full_path,
        decode_times=True,
        chunks=cfg.get("chunks", None),
    )

    present = [v for v in wanted_vars if v in ds.data_vars]
    ds = ds[present]

    if "heights" in ds.dims and "height" not in ds.dims:
        ds = ds.rename({"heights": "height"})

    if "time_10hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_10hz": "time"})
    if "time_20hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_20hz": "time"})
    if "time_60hz" in ds.dims and "time" not in ds.dims:
        ds = ds.rename({"time_60hz": "time"})

    check_dataset_structure(ds)
    return ds


# ============================================================
# CORE PROCESSING (UGUALE A QUELLO CHE FUNZIONA)
# ============================================================

def process_one_file(cfg_path, filename):
    """
    Process a single HF file and produce:
      - stats dataset (window-aggregated)
      - spectra dataset (either raw Welch or log-binned), depending on config.

    Config structure (new)
    ----------------------
    cfg["products"] controls which outputs are produced.

    cfg["averaging"]:
      - window (e.g. "30min")
      - avg_method ("block" or "detrend")
      - butterworth: {enabled, cutoff_seconds, order}

    cfg["spectra"]:
      - welch: {segments, overlap, window, detrend}
      - log_binning: {enabled, N_bin, mode}

    Notes
    -----
    - Stats (pp, uw, etc.) are computed from fluctuations defined in preprocess.get_fluctuations().
    - Spectra are computed from the SAME fluctuations inside spectra_eps().
    - If log-binning is disabled, returned spectra are raw Welch spectra.
    """
    cfg = load_config_json(cfg_path)
    ds = load_single_hf_file(cfg, filename)

    # ------------------------------------------------------------
    # Read config sections (with backward-compatible fallback)
    # ------------------------------------------------------------
    prod = cfg.get("products", {})

    window = cfg["window"]

    spec_cfg = cfg.get("spectra", {})
    welch_cfg = spec_cfg.get("welch", {})
    lb_cfg = spec_cfg.get("log_binning", {})

    welch_segments = int(welch_cfg.get("segments", prod.get("welch_segments", 3)))
    welch_overlap  = float(welch_cfg.get("overlap",  prod.get("welch_overlap", 0.5)))

    lb_enabled = bool(lb_cfg.get("enabled", prod.get("log_binning", True)))
    lb_Nbin    = int(lb_cfg.get("N_bin", prod.get("log_binning_N", 80)))
    lb_mode    = lb_cfg.get("mode", prod.get("log_binning_mode", "area"))  # "area" or "shape"

    # ============================================================
    # BLOCK 0: Preparation (gap filling, rotation)
    # ============================================================
    ds, qc  = fill_gaps(ds, cfg, count_nans=True)
    ds, rot = double_rotation(ds, cfg, return_rotation=True)

    # ============================================================
    # BLOCK 1: Statistics
    # ============================================================
    flux = fluxes_calc(ds, cfg)
    stat = stationarity(ds, cfg)
    stats = xr.merge([flux, stat, rot, qc])

    # Optional physical filter
    if "meanU" in stats:
        stats = stats.where(stats["meanU"] > 0)

    # ============================================================
    # BLOCK 2: Spectra (optional)
    # ============================================================
    spectra_out = None

    if prod.get("spectra", False):

        spectra_welch, epsilon, slopes = spectra_eps(
            ds, cfg,
            Umean=stats["meanU"],
            welch_segments=welch_segments,
            welch_overlap=welch_overlap,
        )

        # Merge spectral diagnostics into stats
        stats = xr.merge([stats, epsilon, slopes])

        # Optional microbarom diagnostics
        if prod.get("microbarom", False):
            mb = get_microbarom(spectra_welch, stats)
            stats = stats.assign(
                MB_fit=mb["area_fit"],
                MB_peak=mb["area_peak"],
                MB_peak_abs=mb["area_peak_abs"],
            )

        # Optional log-binning (if disabled -> raw Welch is returned)
        if lb_enabled:
            spectra_out = bin_spectra_log(spectra_welch, N_bin=lb_Nbin, mode=lb_mode)
        else:
            spectra_out = spectra_welch

    # ============================================================
    # BLOCK 3: Anisotropy (optional)
    # ============================================================
    if prod.get("anisotropy", False):
        aniso = anisotropy_from_stats(stats)
        stats = xr.merge([stats, aniso])
        
    # ============================================================
    # BLOCK 4: Autocorrelation (optional)
    # ============================================================

    return spectra_out, stats





# ============================================================
# FINAL WRAPPER: RUN + SAVE (QUESTO È IL PEZZO NUOVO)
# ============================================================

def run_and_save_one_file(cfg_path, filename, output_dir):
    """
    Run processing on ONE HF file and save outputs.

    Writes:
      <stem>_30min_stats.nc
      <stem>_30min_spectra80.nc  (if spectra enabled)
    """
    os.makedirs(output_dir, exist_ok=True)

    spectra, stats = process_one_file(cfg_path, filename)

    stem = os.path.splitext(filename)[0]

    out_stats = os.path.join(output_dir, f"{stem}_30min_stats.nc")
    stats.to_netcdf(out_stats)
    print("[OK] wrote:", out_stats)

    if spectra is not None:
        out_spec = os.path.join(output_dir, f"{stem}_30min_spectra80.nc")
        spectra.to_netcdf(out_spec)
        print("[OK] wrote:", out_spec)


# ============================================================
# MAIN (USA SOLO QUESTO)
# ============================================================


if __name__ == "__main__":

    CFG = "/Users/mauro_ghirardelli/Documents/windpy/conf/M2HATS_configuration.txt"

    # directories are already in the config, but we read hf_dir explicitly
    with open(CFG, "r") as f:
        cfg = json.load(f)

    HF_DIR  = cfg["paths"]["hf_dir"]
    OUTDIR  = cfg["paths"]["out_dir"]

    # loop over all HF files (must contain 'hf' in the last token)
    for fname in sorted(os.listdir(HF_DIR)):

        if not fname.endswith(".nc"):
            continue

        # IMPORTANT RULE:
        # process only files whose last token contains 'hf'
        # e.g. scp_tc_20121110_18_hf10.nc
        """
        #last_token = os.path.splitext(fname)[0].split("_")[-1]
        #if "hf" not in last_token:
        #    continue
        """
        print("\n" + "=" * 60)
        print("[PROCESSING]", fname)
        print("=" * 60)

        try:
            run_and_save_one_file(CFG, fname, OUTDIR)
        except Exception as e:
            print("[FAILED]", fname)
            print(e)
