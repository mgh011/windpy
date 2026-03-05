#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 08:35:37 2026

@author: mauro_ghirardelli
"""

import xarray as xr


def check_dataset_structure(ds):
    """
    Validate the structure and content of an input xarray.Dataset.

    This function enforces the canonical dataset requirements for the
    pressure–velocity workflow.

    Required dimensions
    -------------------
    - time
    - height
    - tower

    Supported physical variables
    ----------------------------
    At present, the workflow is designed to operate on:
    - P : static pressure
    - u : streamwise velocity component
    - v : cross-stream velocity component
    - w : vertical velocity component

    Rules enforced
    --------------
    - The dataset must contain all required dimensions (time, height, tower).
    - The dataset must contain at least one supported variable among (P, u, v, w).
    - Each supported variable present must:
        * be a DataArray
        * depend on exactly the dimensions {time, height, tower}
          (dimension order may differ)

    Coordinate consistency
    ----------------------
    If multiple supported variables are present, their coordinates along
    (time, height, tower) must be identical.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset to be validated.

    Raises
    ------
    TypeError
        If the input is not an xarray.Dataset.
    ValueError
        If required dimensions are missing.
        If none of the supported variables are present.
        If supported variables do not share the expected dimensional set.
        If supported variables do not share identical coordinates.
    """

    # -----------------------------
    # Type check
    # -----------------------------
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset.")

    # -----------------------------
    # Dimension check
    # -----------------------------
    required_dims = ("time", "height", "tower")
    for dim in required_dims:
        if dim not in ds.dims:
            raise ValueError(f"Missing required dimension: '{dim}'.")

    # -----------------------------
    # Variable check
    # -----------------------------
    supported_vars = ("P", "u", "v", "w")
    present_vars = [v for v in supported_vars if v in ds.data_vars]

    if len(present_vars) == 0:
        raise ValueError(
            "Dataset must contain at least one of the supported variables: "
            f"{supported_vars}."
        )

    # -----------------------------
    # Dimensional consistency check
    # -----------------------------
    expected_dimset = set(required_dims)

    for var in present_vars:
        da = ds[var]

        if not isinstance(da, xr.DataArray):
            raise ValueError(f"Variable '{var}' is not an xarray.DataArray.")

        dimset = set(da.dims)
        if dimset != expected_dimset:
            raise ValueError(
                f"Variable '{var}' has dimensions {da.dims}, "
                f"expected the same set of dimensions {required_dims} "
                "(order may differ)."
            )

    # -----------------------------
    # Coordinate consistency check
    # -----------------------------
    ref_var = present_vars[0]
    ref = ds[ref_var]

    for var in present_vars[1:]:
        da = ds[var]
        for dim in required_dims:
            if not ref.coords[dim].identical(da.coords[dim]):
                raise ValueError(
                    f"Coordinate mismatch in dimension '{dim}' "
                    f"between variables '{ref_var}' and '{var}'."
                )

    return None

