"""
Library Features:
Utilities for managing GRISO interpolation of rain gauge data to a regular geographic grid

Name:          griso_interpolator
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

from .griso_preprocessing import prepare_grid, prepare_stations
from .griso_correlation import build_kernels
from prism.common.data_utils import mask_nodata, metadata_nodata, DEFAULT_NODATA

LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class GrisoConfig:
    """Correlation, kernel and missing-data settings for rainfall interpolation."""
    radius_km: float = 30.0
    correlation: str = 'fixed'
    min_radius_km: float = 5.0
    station_nodata: tuple[float, ...] = DEFAULT_NODATA
    background_nodata: tuple[float, ...] = DEFAULT_NODATA
    dynamic_min_valid_background: float = 0.0
    dynamic_wet_gauge_threshold: float = 0.2
    dynamic_valid_pixels_factor: float = 100.0
    kernel_cutoff: float = 0.001

    def __post_init__(self) -> None:
        if self.radius_km <= 0:
            raise ValueError('radius_km must be positive')
        if self.correlation not in ('fixed', 'dynamic'):
            raise ValueError("correlation must be 'fixed' or 'dynamic'")
        if not (0 < self.min_radius_km < self.radius_km):
            raise ValueError('min_radius_km must be positive and less than radius_km')
        if self.dynamic_valid_pixels_factor <= 0:
            raise ValueError('dynamic_valid_pixels_factor must be positive')
        if self.kernel_cutoff < 0:
            raise ValueError('kernel_cutoff must be nonnegative')


class GrisoInterpolator:
    """GRISO rain gauge interpolation to a regular geographic (EPSG:4326) grid.

    Example::
        interpolator = GrisoInterpolator(GrisoConfig(radius_km=30))
        rainfall = interpolator.interpolate(stations, target_grid)

    In dynamic mode, the background estimates the radius of each local
    correlation kernel. Rainfall values are interpolated from the stations.
    """
    def __init__(self, config: GrisoConfig | dict | None = None):
        if config is None:
            config = GrisoConfig()
        if isinstance(config, dict):
            config = GrisoConfig(**config)
        if not isinstance(config, GrisoConfig):
            raise TypeError('config must be GrisoConfig or a config dictionary')
        self.config = config

    def interpolate(self, stations: pd.DataFrame, grid: xr.DataArray,
                    background: xr.DataArray | None = None, *,
                    station_nodata=None, background_nodata=None) -> xr.DataArray:
        """Interpolate station rainfall and preserve the target grid orientation."""
        geom = prepare_grid(grid, self.config.radius_km)
        station_missing = self.config.station_nodata if station_nodata is None else station_nodata
        st = prepare_stations(stations, geom, nodata=station_missing)
        LOG.info('GRISO mode=%s radius_km=%.2f stations=%d',
                 self.config.correlation, self.config.radius_km, len(st))
        bg = None
        if self.config.correlation == 'dynamic':
            if background is None:
                raise ValueError('Dynamic GRISO needs background on the target grid')
            if not isinstance(background, xr.DataArray):
                raise TypeError('background must be an xarray.DataArray')
            if set(background.dims) != {'lat','lon'} or background.ndim != 2:
                raise ValueError('background must be a 2D array with lat/lon dimensions')
            bg_aligned = background.transpose('lat','lon')
            # Background and target must refer to exactly the same grid cells.
            if not (np.array_equal(bg_aligned.lon.values, geom.lon)
                    and np.array_equal(bg_aligned.lat.values, geom.lat)):
                raise ValueError('Background lat/lon coordinates must match target grid exactly')
            background_missing = self.config.background_nodata if background_nodata is None else background_nodata
            bg = mask_nodata(bg_aligned.values, background_missing,
                             metadata=metadata_nodata(bg_aligned))

        kernels = build_kernels(st, geom, self.config.radius_km,
                                self.config.min_radius_km,
                                self.config.correlation, bg,
                                min_valid_background=self.config.dynamic_min_valid_background,
                                wet_gauge_threshold=self.config.dynamic_wet_gauge_threshold,
                                valid_pixels_factor=self.config.dynamic_valid_pixels_factor,
                                kernel_cutoff=self.config.kernel_cutoff)
        n = len(kernels)
        pad = geom.radius_cells
        # Each column samples one station's kernel at all station locations.
        A = np.eye(n, dtype=float)
        for j, kernel in enumerate(kernels):
            for i, other in enumerate(kernels):
                dy = other.station_row - kernel.station_row
                dx = other.station_col - kernel.station_col
                if abs(dy) <= pad and abs(dx) <= pad:
                    A[i, j] = kernel.values[pad+dy, pad+dx]
        gauge_values = st['value'].to_numpy(dtype=float)
        # Least squares also handles rank-deficient station correlation systems.
        weights, _, rank, _ = np.linalg.lstsq(A, gauge_values, rcond=None)
        if rank < n:
            LOG.warning('Correlation system is rank deficient (%d of %d); least-squares used', rank, n)

        # Sum weighted kernels on a padded grid, then crop and remove negatives.
        out = np.zeros((len(geom.lat)+2*pad, len(geom.lon)+2*pad), dtype=float)
        for weight, kernel in zip(weights, kernels):
            y, x = kernel.station_row, kernel.station_col
            out[y:y+2*pad+1, x:x+2*pad+1] += weight * kernel.values
        out = np.clip(out[pad:-pad, pad:-pad], 0, None)
        result = xr.DataArray(out, dims=('lat','lon'),
                              coords={'lat':geom.lat.copy(), 'lon':geom.lon.copy()},
                              name='precip')
        result.attrs = dict(grid.attrs)
        result.attrs.update({'method':'GRISO', 'radius_km':self.config.radius_km,
                             'correlation':self.config.correlation,
                             'n_stations':n})
        return result.transpose(*grid.dims)
