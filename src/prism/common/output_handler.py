"""Write GRISO products to GeoTIFF, Arc ASCII or NetCDF."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import xarray as xr
from .time_utils import resolve_path


class OutputHandler:
    def __init__(self, config: dict, domain: str | None = None):
        self.config = config
        self.domain = domain

    def path(self, time_step=None) -> Path:
        return Path(resolve_path(self.config['file'], time_step, self.domain))

    def write(self, data: xr.DataArray, time_step=None, overwrite: bool = False) -> Path:
        path = self.path(time_step)
        if path.exists() and not overwrite:
            raise FileExistsError(f'{path} already exists; enable overwrite to replace it')
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        nodata = float(self.config.get('nodata', -9999.0))
        if not np.isfinite(nodata):
            raise ValueError('Output nodata must be finite')
        if ext in ('.nc','.nc4'):
            output = data.copy(deep=False)
            if time_step is not None:
                output = output.expand_dims(time=[pd.Timestamp(time_step)])
            output.attrs = dict(output.attrs)
            # Store missing values through CF encoding; the raster CRS is not a CF variable.
            output.attrs.pop('crs',None)
            output.to_netcdf(path, encoding={(output.name or 'precip'): {'_FillValue': nodata}})
        elif ext in ('.tif','.tiff','.asc'):
            arr = data.transpose('lat','lon')
            lon, lat = np.asarray(arr.lon.values), np.asarray(arr.lat.values)
            if lon.size < 2 or lat.size < 2:
                raise ValueError('Raster output needs at least 2 longitude and latitude coordinates')
            # Rasterio always writes from upper left; orient independently of input.
            if lon[0] > lon[-1]:
                arr = arr.isel(lon=slice(None,None,-1))
            if lat[0] < lat[-1]:
                arr = arr.isel(lat=slice(None,None,-1))
            lon, lat = np.asarray(arr.lon.values), np.asarray(arr.lat.values)
            dx, dy = float(abs(lon[1]-lon[0])), float(abs(lat[1]-lat[0]))
            # Coordinates locate cell centres; the raster transform starts at the outer corner.
            transform = from_origin(float(lon[0]-dx/2),float(lat[0]+dy/2),dx,dy)
            profile = {'driver': 'GTiff' if ext!='.asc' else 'AAIGrid',
                       'height':arr.shape[0],'width':arr.shape[1],
                       'count':1,'dtype':'float32', 'crs':'EPSG:4326',
                       'transform':transform, 'nodata':nodata}
            values = np.asarray(arr.values, dtype='float32')
            values = np.where(np.isfinite(values),values,nodata)
            with rasterio.open(path,'w',**profile) as dst:
                dst.write(values,1)
        else:
            raise ValueError(f'Unsupported output extension: {ext}')
        return path
