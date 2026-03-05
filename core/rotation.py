#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 09:08:13 2026

@author: mauro_ghirardelli
"""

# windy/core/rotation.py

import numpy as np
import xarray as xr


def double_rotation(ds, config, return_rotation=True):
    """
    Double rotation of wind components.

    Uses block means over config["window"] to compute:
      - theta = atan2(v̄, ū)  (horizontal rotation)
      - phi   = atan2(w̄, sqrt(ū²+v̄²)) (tilt rotation)

    Then applies rotations to instantaneous u,v,w.

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain variables: u, v, w.
    config : dict
        Must contain: "window" (e.g. "10min")
    return_rotation : bool
        If True returns (ds_rot, rotation_ds), else only ds_rot.

    Returns
    -------
    ds_rot : xarray.Dataset
        Dataset with rotated u,v,w.
    rotation : xarray.Dataset (optional)
        Variables: dir (deg), theta (rad), phi (rad) on resampled time grid.
    """
    window = config["window"]

    for v in ("u", "v", "w"):
        if v not in ds.data_vars:
            raise ValueError(f"double_rotation requires variable '{v}' in dataset.")

    # block means
    ds_mean = ds.resample(time=window).mean()

    # angles
    theta = np.arctan2(ds_mean["v"], ds_mean["u"])
    phi = np.arctan2(ds_mean["w"], np.sqrt(ds_mean["u"] ** 2 + ds_mean["v"] ** 2))

    rotation = xr.Dataset(
        data_vars=dict(
            dir=(270 - theta * 180 / np.pi) % 360,
            theta=theta,
            phi=phi,
        )
    )

    # broadcast trig terms back to original time
    ct = np.cos(theta).reindex(time=ds.time).ffill("time")
    st = np.sin(theta).reindex(time=ds.time).ffill("time")
    cp = np.cos(phi).reindex(time=ds.time).ffill("time")
    sp = np.sin(phi).reindex(time=ds.time).ffill("time")

    # rotate
    u = ds["u"]
    v = ds["v"]
    w = ds["w"]

    u_rot = ct * cp * u + st * cp * v + sp * w
    v_rot = -st * u + ct * v
    w_rot = -ct * sp * u - st * sp * v + cp * w

    ds_rot = ds.assign(u=u_rot, v=v_rot, w=w_rot)

    if return_rotation:
        return ds_rot, rotation
    return ds_rot
