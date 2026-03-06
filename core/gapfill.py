# windy/core/gapfill.py

import warnings
import numpy as np
import xarray as xr

import warnings
import numpy as np
import xarray as xr


def fill_gaps(ds, config, count_nans=True):
    """
    Gap filling by block:
    - replace inf with nan
    - interpolate short gaps along time
    - fill residual nans with block mean
    - final fallback to zero
    """
    window = config["window"]

    gap_cfg = config.get("gapfill", {})
    method = gap_cfg.get("method", config.get("gap_filling", "interp"))

    if method != "interp":
        warnings.warn(
            f"Gap filling method {method} not recognized, will use interpolation"
        )
        method = "interp"

    var_list = list(ds.data_vars)
    ds_groups = ds.resample(time=window)

    nans = []
    gap_fill = []
    nan_warned = False

    for _, group in ds_groups:
        group = group.copy()

        # --- count nans / infs before filling ---
        nan_array = ~np.isfinite(group[var_list[0]])
        for var in var_list[1:]:
            nan_array = nan_array | ~np.isfinite(group[var])
        nans.append(nan_array)

        # --- replace infs with nan ---
        for var in var_list:
            group[var] = group[var].where(np.isfinite(group[var]), other=np.nan)

        # --- interpolate short gaps ---
        if method == "interp":
            group = group.interpolate_na(dim="time", limit=10)

        # --- residual nans ---
        for var in var_list:
            if bool(group[var].isnull().any()):
                # fill with block mean
                block_mean = group[var].mean(dim="time", skipna=True)
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=block_mean
                )

                # if still nan (e.g. whole block invalid), fill with zero
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=0
                )

                if not nan_warned:
                    warnings.warn(
                        f"Nans survived gap filling in time period "
                        f"{group.time[0].values}, set to mean/0. Will warn only once."
                    )
                    nan_warned = True

        gap_fill.append(group)

    ds_filled = xr.concat(gap_fill, dim="time")

    if count_nans:
        nans = xr.concat(nans, dim="time")
        nan_perc = (
            nans.resample(time=window).sum(dim="time")
            / nans.resample(time=window).count(dim="time")
        ).rename("QCnan")
        return ds_filled, nan_perc
    else:
        return ds_filled



    
 
#---- old code - troubleshooting
    
def fill_gaps_old(ds, config, count_nans=True):
    """
    Original gap filling routine (kept intentionally identical in logic).

    Only changes:
    - use numpy.isfinite instead of xr.ufuncs.isfinite
    - fix QCnan bookkeeping to avoid scalar time conflict
    """

    window = config["window"]
    
    # gap filling method: support both new-style and legacy keys
    gap_cfg = config.get("gapfill", {})
    method = gap_cfg.get("method", config.get("gap_filling", "interp"))


    if method != "interp":
        warnings.warn(
            f"Gap filling method {method} not recognized, will use interpolation"
        )
        method = "interp"

    var_list = list(ds.data_vars)
    ds_groups = ds.resample(time=window)

    nans = []
    gap_fill = []
    nan_warned = False

    for _, group in ds_groups:
        # --- count nans (OR over variables) ---
        nan_array = ~np.isfinite(group[var_list[0]])
        for var in var_list[1:]:
            nan_array = nan_array | ~np.isfinite(group[var])
        nans.append(nan_array)

        # --- remove infs ---
        for var in var_list:
            group[var] = group[var].where(np.isfinite(group[var]), other=np.nan)

        # --- gap filling ---
        if method == "interp":
            group = group.interpolate_na(dim="time", limit=10)

        # --- residual nans ---
        for var in var_list:
            if group[var].isnull().any():
                # fill with block mean
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=group[var].mean(dim="time")
                )
                # if still nan (empty block), fill with zero
                group[var] = group[var].where(
                    np.isfinite(group[var]), other=0
                )

                if not nan_warned:
                    warnings.warn(
                        f"Nans survived gap filling in time period "
                        f"{group.time[0].values}, set to mean/0. Will warn only once."
                    )
                    nan_warned = True

        gap_fill.append(group)

    # --- reconcat ---
    ds_filled = xr.concat(gap_fill, dim="time")

    # --- QCnan ---
    nans = xr.concat(nans, dim="time")
    nan_perc = (
        nans.resample(time=window)
        .sum(dim="time") / nans.resample(time=window).count(dim="time")
    ).rename("QCnan")

    if count_nans:
        return ds_filled, nan_perc
    else:
        return ds_filled

