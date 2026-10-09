"""
Library Features:
Utilities for managing spatial and geographic data

Name:          geo_utils
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""
import numpy as np
from scipy.optimize import curve_fit


def deg2km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance, in kilometres (Fixed earth radius 6372.8 km)."""
    lat1, lat2 = np.radians([lat1, lat2])
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * 6372.8 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def spherical_covariance(distance: np.ndarray, radius: float) -> np.ndarray:
    """Spherical covariance with zero nugget and unit sill."""
    d = np.asarray(distance, dtype=float) / radius
    return np.where(d < 1, 1 - 1.5 * d + 0.5 * d**3, 0.0)


def fit_spherical(distance: np.ndarray, sample: np.ndarray, min_radius: float,
                  max_radius: float, start_radius: float) -> tuple[np.ndarray, float]:
    """Fit a bounded covariance radius; use min_radius when fitting is unavailable."""
    mask = np.isfinite(sample) & np.isfinite(distance)
    if mask.sum() < 3 or not np.any(sample[mask] > 0):
        return spherical_covariance(distance, min_radius), min_radius
    try:
        fitted, _ = curve_fit(
            spherical_covariance, distance[mask], sample[mask],
            p0=[np.clip(start_radius, min_radius, max_radius)],
            bounds=([min_radius], [max_radius]), maxfev=2000,
        )
        radius = float(fitted[0])
        return spherical_covariance(distance, radius), radius
    except (ValueError, RuntimeError, FloatingPointError):
        return spherical_covariance(distance, min_radius), min_radius


def geocentric_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit-sphere coordinates for longitude/latitude nearest-neighbour mapping."""
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
