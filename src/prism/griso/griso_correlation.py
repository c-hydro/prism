"""
Library Features:
Utilities for managing GRISO station-centred spherical correlation kernels (fixed and dynamic)

Name:          griso_correlation
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import correlate
from prism.common.geo_utils import spherical_covariance, fit_spherical
from .griso_preprocessing import PreparedGrid

@dataclass(frozen=True)
class LocalKernel:
    """Correlation values and fitted radius centred on one station's grid cell."""
    station_row: int
    station_col: int
    values: np.ndarray
    fitted_radius_km: float


def _pair_count_window(mask: np.ndarray, effective_step: float, radius_km: float) -> np.ndarray:
    """Count spatial neighbours used to normalize the local autocorrelation."""
    yarr, xarr = np.indices(mask.shape)
    pair_counts = np.zeros(mask.shape, dtype=float)
    for y, x in np.argwhere(mask):
        r = np.hypot(xarr - y, yarr - x) * effective_step
        pair_counts += (mask & (r > 0) & (r <= radius_km))
    # Cells without neighbours need a nonzero divisor during normalization.
    pair_counts[pair_counts == 0] = -1.0
    return pair_counts


def build_kernels(stations, geom: PreparedGrid, radius_km: float,
                  min_radius_km: float, mode: str,
                  background: np.ndarray | None = None, *,
                  min_valid_background: float = 0.0,
                  wet_gauge_threshold: float = 0.2,
                  valid_pixels_factor: float = 100.0,
                  kernel_cutoff: float = 0.001) -> list[LocalKernel]:
    """Build a fixed or background-fitted correlation kernel for each station."""
    pad = geom.radius_cells
    offsets = np.arange(-pad, pad + 1)
    x, y = np.meshgrid(offsets, offsets)
    dist = np.hypot(x, y) * geom.effective_cellsize_km
    fixed = spherical_covariance(dist, radius_km)
    short = spherical_covariance(dist, min_radius_km)
    rs_ext = None
    pair_counts = None
    if mode == 'dynamic':
        if background is None:
            raise ValueError("Dynamic correlation requires an auxiliary gridded background")
        if background.shape != geom.grid.shape:
            raise ValueError("Background shape differs from target grid")
        # Padding keeps local windows the same size at the grid boundaries.
        rs_ext = np.pad(background.astype(float), pad, constant_values=np.nan)
        pair_counts = _pair_count_window(dist <= radius_km, geom.effective_cellsize_km, radius_km)
        valid_pixels_threshold = valid_pixels_factor / geom.effective_cellsize_km
    kernels = []
    for row in stations.itertuples(index=False):
        center_r, center_c = int(row.row), int(row.col)
        values, fit_radius = fixed.copy(), radius_km
        if mode == 'dynamic':
            window = rs_ext[center_r:center_r+2*pad+1, center_c:center_c+2*pad+1].copy()
            window[dist > radius_km] = np.nan
            valid = window[window >= min_valid_background]
            fully_valid = len(valid) == np.count_nonzero(~np.isnan(window))
            uniform = len(valid) > 0 and np.all(valid == window[pad, pad])
            if (fully_valid and uniform) or np.nansum(window) == 0:
                pass  # Keep the full radius for uniform or zero-total backgrounds.
            elif len(valid) < valid_pixels_threshold and row.value > wet_gauge_threshold:
                pass  # Keep the full radius for wet gauges with few valid pixels.
            elif len(valid) > valid_pixels_threshold and np.nanvar(window) > 0:
                norm = np.nan_to_num(window - np.nanmean(window), nan=0.0)
                autocorr = correlate(norm, norm, mode='full', method='auto')
                # Crop to the kernel window and normalize by pair count and variance.
                sample = autocorr[pad:-pad, pad:-pad] / pair_counts / np.nanvar(window)
                sample = np.maximum(sample, 0)
                sample[dist > radius_km] = 0
                values, fit_radius = fit_spherical(
                    dist, sample, min_radius_km, radius_km,
                    max(min_radius_km, radius_km - geom.effective_cellsize_km))
            else:
                # Use the minimum radius when neither a full-radius fallback nor a fit applies.
                values, fit_radius = short.copy(), min_radius_km
        # Discard small correlation values before assembling the station system.
        values = np.where(values >= kernel_cutoff, values, 0)
        kernels.append(LocalKernel(center_r, center_c, values, fit_radius))
    return kernels
