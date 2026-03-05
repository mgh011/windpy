#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 10:38:14 2026

@author: mauro_ghirardelli
"""
import numpy as np
import xarray as xr


def anisotropy_from_stats(stats):
    xb, yb, rgb = barycentric_anisotropy_xr(stats, return_rotation=False)

    # rename rgb dim to channel, and variable to rgb_vec
    rgb = rgb.rename({"rgb": "channel"}).assign_coords(channel=["R", "G", "B"])
    rgb.name = "rgb_vec"

    out = xr.Dataset(data_vars=dict(xb=xb, yb=yb, rgb_vec=rgb))
    out["xb"].attrs.update(long_name="barycentric anisotropy x-coordinate")
    out["yb"].attrs.update(long_name="barycentric anisotropy y-coordinate")
    out["rgb_vec"].attrs.update(long_name="barycentric anisotropy RGB weights", dims="channel")
    return out


# helper: generic function that works on xarray 
def barycentric_anisotropy_xr(ds, return_rotation=False):
    """
    Vectorized barycentric + RGB anisotropy from Reynolds stresses.
    Required vars: uu,vv,ww,uv,uw,vw
    Works with any dims (e.g. time,tower,height).

    Returns:
      xb, yb : DataArray with same dims as uu
      rgb    : DataArray with extra dim 'rgb' = ['R','G','B']
      optionally eigvals (dim 'eig'), eigvecs (dims 'i','eig')
    """
    required = ["uu", "vv", "ww", "uv", "uw", "vw"]
    missing = [v for v in required if v not in ds]
    if missing:
        raise KeyError(f"Missing vars for anisotropy: {missing}")

    # NaN-safe like your original
    ds0 = ds[required].where(np.isfinite(ds[required]), other=0.0)

    uu, vv, ww = ds0["uu"], ds0["vv"], ds0["ww"]
    uv, uw, vw = ds0["uv"], ds0["uw"], ds0["vw"]

    trace = uu + vv + ww

    def norm(x):
        r = xr.where(trace != 0, x / trace, 0.0)
        r = xr.where(np.isfinite(r), r, 0.0)
        return r

    # deviatoric anisotropy tensor b_ij (symmetric)
    b11 = norm(uu) - 1.0/3.0
    b22 = norm(vv) - 1.0/3.0
    b33 = norm(ww) - 1.0/3.0
    b12 = norm(uv)
    b13 = norm(uw)
    b23 = norm(vw)

    # Pack into matrix with dims (i,j,...)
    i = xr.DataArray([0, 1, 2], dims="i")
    j = xr.DataArray([0, 1, 2], dims="j")

    B = xr.zeros_like(b11).expand_dims(i=3, j=3).copy()
    B.loc[dict(i=0, j=0)] = b11
    B.loc[dict(i=1, j=1)] = b22
    B.loc[dict(i=2, j=2)] = b33
    B.loc[dict(i=0, j=1)] = b12
    B.loc[dict(i=1, j=0)] = b12
    B.loc[dict(i=0, j=2)] = b13
    B.loc[dict(i=2, j=0)] = b13
    B.loc[dict(i=1, j=2)] = b23
    B.loc[dict(i=2, j=1)] = b23

    # Eigendecomposition over all other dims
    eigvals, eigvecs = xr.apply_ufunc(
        np.linalg.eigh,
        B,
        input_core_dims=[["i", "j"]],
        output_core_dims=[["eig"], ["i", "eig"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float],
    )
    # eigh returns ascending eigenvalues along 'eig'. We want descending.
    # We do argsort + take_along_axis, vectorized.

    idx = xr.apply_ufunc(
        np.argsort,
        eigvals,
        input_core_dims=[["eig"]],
        output_core_dims=[["eig"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    )[..., ::-1]  # descending

    eigvals_s = xr.apply_ufunc(
        np.take_along_axis,
        eigvals,
        idx,
        kwargs={"axis": -1},
        input_core_dims=[["eig"], ["eig"]],
        output_core_dims=[["eig"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # For eigvecs: shape (..., i, eig). Need idx broadcast to (..., i, eig)
    idx_i = idx.broadcast_like(eigvecs.isel(i=0))  # (..., eig)
    idx_i = idx_i.expand_dims(i=3).transpose(..., "i", "eig")  # (..., i, eig)

    eigvecs_s = xr.apply_ufunc(
        np.take_along_axis,
        eigvecs,
        idx_i,
        kwargs={"axis": -1},
        input_core_dims=[["i", "eig"], ["i", "eig"]],
        output_core_dims=[["i", "eig"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Enforce trace=0 numerical safety
    eigvals_s = eigvals_s - eigvals_s.mean("eig")

    # l1>=l2>=l3
    l1 = eigvals_s.isel(eig=0)
    l2 = eigvals_s.isel(eig=1)
    l3 = eigvals_s.isel(eig=2)

    xb = (l1 - l2) + 1.5*l3 + 0.5
    yb = (np.sqrt(3)/2.0) * (3.0*l3 + 1.0)
    xb.name = "xb"
    yb.name = "yb"

    # RGB mapping
    R = (l1 - l2)
    Bc = 2.0*(l2 - l3)
    G = (3.0*l3 + 1.0)

    rgb = xr.concat([R, G, Bc], dim="rgb")
    rgb = rgb.assign_coords(rgb=["R", "G", "B"]).clip(0.0, 1.0)
    rgb.name = "rgb"

    if return_rotation:
        eigvals_s.name = "eigvals"
        eigvecs_s.name = "eigvecs"
        return xb, yb, rgb, eigvals_s, eigvecs_s

    return xb, yb, rgb