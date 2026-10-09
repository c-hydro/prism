"""
Library Features:
Utilities for managing data and handling nodata values

Name:          data_utils
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""

from collections.abc import Iterable
import numpy as np

DEFAULT_NODATA = (-9999.0,)


def nodata_values(values=None, *, default=DEFAULT_NODATA) -> tuple[float, ...]:
    """Normalize a number or sequence; None uses default and [] disables it."""
    if values is None:
        values = default
    if np.isscalar(values):
        values = (values,)
    elif not isinstance(values, Iterable):
        raise TypeError('nodata must be a number, a sequence or None')
    result = []
    for value in values:
        try:
            result.append(float(value))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid nodata value: {value!r}') from exc
    return tuple(result)


def metadata_nodata(array) -> tuple[float, ...]:
    """Find CF-style nodata metadata on an xarray DataArray (if available)."""
    found = []
    for container in (getattr(array, 'attrs', {}), getattr(array, 'encoding', {})):
        for name in ('_FillValue', 'missing_value', 'nodata'):
            if name in container and container[name] is not None:
                found.extend(nodata_values(container[name], default=()))
    return tuple(found)


def mask_nodata(data, nodata=None, *, metadata=()) -> np.ndarray:
    """Convert missing sentinels and nonfinite numeric values to NaN."""
    values = np.asarray(data, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    # Exact matching preserves valid values close to a configured sentinel.
    for sentinel in (*nodata_values(nodata), *nodata_values(metadata, default=())):
        if np.isfinite(sentinel):
            values[values == sentinel] = np.nan
    return values
