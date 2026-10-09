"""
Library Features:
Utilities for managing GRISO preprocessing of grids and rain gauge stations

Name:          griso_preprocessing
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from prism.common.geo_utils import deg2km, geocentric_xyz
from prism.common.data_utils import mask_nodata

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class PreparedGrid:
    lon: np.ndarray
    lat: np.ndarray
    grid: xr.DataArray
    cellsize_km: float
    radius_cells: int
    effective_cellsize_km: float


def prepare_grid(grid: xr.DataArray, radius_km: float) -> PreparedGrid:
    """Validate grid coordinates and determine the kernel radius in cells."""
    if not isinstance(grid, xr.DataArray):
        raise TypeError("grid must be an xarray.DataArray with 1D lon and lat coordinates")
    if not {'lon', 'lat'}.issubset(grid.coords):
        raise ValueError("grid requires 'lon' and 'lat' coordinates")
    lon, lat = np.asarray(grid.lon.values, dtype=float), np.asarray(grid.lat.values, dtype=float)
    if lon.ndim != 1 or lat.ndim != 1 or min(lon.size, lat.size) < 2:
        raise ValueError("grid must have 1D lon/lat coordinates, at least 2 each")
    if set(grid.dims) != {'lat', 'lon'} or grid.ndim != 2:
        raise ValueError("grid must be two-dimensional with dims lat and lon")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("grid coordinates contain invalid values")
    if not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)):
        raise ValueError("longitude must be strictly monotonic")
    if not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)):
        raise ValueError("latitude must be strictly monotonic")
    if not np.allclose(abs(np.diff(lon)), abs(lon[1]-lon[0]), rtol=1e-3):
        raise ValueError("longitude spacing must be regular")
    if not np.allclose(abs(np.diff(lat)), abs(lat[1]-lat[0]), rtol=1e-3):
        raise ValueError("latitude spacing must be regular")
    cellsize = deg2km(lat[0], lon[0], lat[1], lon[0])
    if cellsize <= 0:
        raise ValueError("invalid grid resolution")
    # Use an even radius in cells and adjust the kilometre step to match radius_km.
    cells = max(2, int(np.rint(radius_km / cellsize)))
    if cells % 2:
        cells += 1
    effective = radius_km / cells
    arr = grid.transpose('lat', 'lon')
    return PreparedGrid(lon=lon, lat=lat, grid=arr,
                        cellsize_km=cellsize, radius_cells=cells,
                        effective_cellsize_km=effective)


def prepare_stations(stations: pd.DataFrame, geom: PreparedGrid, nodata=None) -> pd.DataFrame:
    """Filter invalid/remote gauges and average multiple gauges in the same cell."""
    if not isinstance(stations, pd.DataFrame):
        raise TypeError("stations must be a pandas.DataFrame")
    for col in ('lon', 'lat', 'value'):
        if col not in stations.columns:
            raise ValueError(f"stations missing required '{col}' column")
    rows = stations.copy()
    for col in ('lon', 'lat', 'value'):
        rows[col] = pd.to_numeric(rows[col], errors='coerce')
    rows['value'] = mask_nodata(rows['value'], nodata)
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=['lon','lat','value'])
    rows = rows.loc[rows['value'] >= 0].copy()
    if rows.empty:
        raise ValueError("No valid nonnegative station observations")
    # Include stations up to half a cell beyond the outer grid coordinates.
    hlon, hlat = abs(geom.lon[1]-geom.lon[0])/2, abs(geom.lat[1]-geom.lat[0])/2
    in_box = (rows.lon.between(geom.lon.min()-hlon, geom.lon.max()+hlon)
              & rows.lat.between(geom.lat.min()-hlat, geom.lat.max()+hlat))
    if (~in_box).any():
        LOG.warning("Ignoring %d station(s) outside grid extent", int((~in_box).sum()))
    rows = rows.loc[in_box].copy()
    if rows.empty:
        raise ValueError("No station observations within the target grid extent")

    x, y = np.meshgrid(geom.lon, geom.lat)
    # Unit-sphere coordinates let the tree compare geographic distances.
    tree = cKDTree(geocentric_xyz(x.ravel(), y.ravel()))
    _, idx = tree.query(geocentric_xyz(rows.lon.to_numpy(), rows.lat.to_numpy()))
    rows['row'], rows['col'] = np.unravel_index(idx, x.shape)
    if rows.groupby(['row','col']).size().gt(1).any():
        LOG.info("Averaging gauges sharing the same target grid cell")
    # A cell contributes one observation, averaged across all its stations.
    result = rows.groupby(['row','col'], as_index=False, sort=True)['value'].mean()
    return result
