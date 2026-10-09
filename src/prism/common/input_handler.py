"""
Library Features:
Station and geographic raster readers for GRISO.

Name:          input_handler
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from .time_utils import resolve_path
from .data_utils import mask_nodata, metadata_nodata

LOG = logging.getLogger(__name__)


class InputHandler:
    def __init__(self, config: dict, domain: str | None = None):
        self.config = config
        self.domain = domain
        self._cache: dict[str, pd.DataFrame] = {}

    def _path(self, template: str, time_step=None, **tags: str) -> Path:
        return Path(resolve_path(template, time_step, self.domain, **tags))

    @staticmethod
    def _csv(file: Path, cfg: dict) -> pd.DataFrame:
        return pd.read_csv(file, sep=cfg.get('delimiter', ','),
                           header=0 if cfg.get('header', True) else None)

    @staticmethod
    def _normalize(frame: pd.DataFrame, cfg: dict, fields: list[str]) -> pd.DataFrame:
        """Columns mapping maps canonical names to source header or column index."""
        mapping = cfg.get('columns', {})
        renames = {}
        for name in fields:
            if name in mapping:
                source = mapping[name]
                if isinstance(source, str) and source.isdigit() and source not in frame.columns:
                    source = int(source)
                if source not in frame.columns:
                    raise KeyError(f"Column {source!r} for {name!r} missing from input")
                renames[source] = name
        return frame.rename(columns=renames)

    def _coords(self, settings: dict, time_step) -> pd.DataFrame:
        cfg = settings['coordinates']
        filename = self._path(cfg['file'], time_step)
        cache_key = f'coords:{filename}'
        if cache_key not in self._cache:
            data = self._normalize(self._csv(filename, cfg), cfg, ['code','lat','lon'])
            needed = {'code','lat','lon'}
            if not needed.issubset(data.columns):
                raise ValueError(f"Coordinates file needs columns {needed}")
            self._cache[cache_key] = data[['code','lat','lon']].copy()
        return self._cache[cache_key]

    @staticmethod
    def _clean(stations: pd.DataFrame, nodata=None) -> pd.DataFrame:
        needed = {'lon','lat','value'}
        if not needed.issubset(stations.columns):
            raise ValueError(f'Stations data needs columns {needed}')
        result = stations.copy()
        if 'code' not in result:
            result['code'] = np.arange(len(result)).astype(str)
        for col in ('lon','lat','value'):
            result[col] = pd.to_numeric(result[col], errors='coerce')
        result['value'] = mask_nodata(result['value'], nodata)
        result = result.dropna(subset=['lon','lat','value'])
        result = result.loc[result['value'] >= 0].copy()
        return result[['code','lon','lat','value']].reset_index(drop=True)

    def read_stations(self, time_step) -> pd.DataFrame:
        cfg = self.config['stations']
        mode = cfg.get('type', 'timestep')
        if mode == 'timestep':
            path = self._path(cfg['file'], time_step)
            data = self._normalize(self._csv(path, cfg), cfg, ['code','lon','lat','value'])
            if 'coordinates' in cfg:
                if 'code' not in data:
                    raise ValueError('Timestep station file needs station code when coordinates are external')
                data = data.merge(self._coords(cfg, time_step), on='code', how='left', validate='many_to_one')
        elif mode == 'timeseries':
            layout = cfg.get('layout', 'wide')
            if layout == 'per_station':
                coords = self._coords(cfg, time_step)
                items = []
                for sta in coords.itertuples(index=False):
                    path = self._path(cfg['file'], time_step, code=str(sta.code), station_name=str(sta.code))
                    if not path.exists():
                        LOG.warning('Station timeseries missing: %s', path)
                        continue
                    data = self._normalize(self._csv(path, cfg), cfg, ['time','value'])
                    if 'value' not in data:
                        raise ValueError('Per-station input requires value column')
                    record = self._select_row(data, cfg, time_step)
                    value = float(record['value']) if not record.empty else float('nan')
                    if pd.notna(value):
                        items.append({'code':sta.code,'lon':sta.lon,'lat':sta.lat,'value':value})
                data = pd.DataFrame(items, columns=['code','lon','lat','value'])
            else:
                path = self._path(cfg['file'], time_step)
                cache_key = f'timeseries:{path}'
                if cache_key not in self._cache:
                    # Reuse the full series while processing successive timesteps.
                    self._cache[cache_key] = self._csv(path, cfg)
                data = self._cache[cache_key].copy()
                if layout == 'wide':
                    data = self._normalize(data, cfg, ['time'])
                    record = self._select_row(data, cfg, time_step)
                    if record.empty:
                        data = pd.DataFrame(columns=['code','lon','lat','value'])
                    else:
                        vals = record.drop(labels=['time']).rename_axis('code').reset_index(name='value')
                        data = vals.merge(self._coords(cfg, time_step), on='code', how='left')
                elif layout == 'long':
                    data = self._normalize(data, cfg, ['time','code','lon','lat','value'])
                    # Exclude missing and negative rainfall before temporal aggregation.
                    data['value'] = mask_nodata(pd.to_numeric(data['value'],errors='coerce'), cfg.get('nodata'))
                    data.loc[data['value'] < 0, 'value'] = np.nan
                    data = self._subset_time(data, cfg, time_step)
                    if 'coordinates' in cfg:
                        data = data.merge(self._coords(cfg, time_step), on='code', how='left', validate='many_to_one')
                else:
                    raise ValueError("timeseries.layout must be 'wide', 'long' or 'per_station'")
        else:
            raise ValueError("stations.type must be 'timestep' or 'timeseries'")
        result = self._clean(data, cfg.get('nodata'))
        LOG.info('Read %d station(s) for %s', len(result), time_step)
        return result

    @staticmethod
    def _select_time(data: pd.DataFrame, cfg: dict, time_step) -> pd.DataFrame:
        """Select an exact timestep or the right-labelled window (t-frequency, t]."""
        if 'time' not in data:
            raise ValueError('Time series input requires time column')
        timestamps = pd.to_datetime(data['time'], errors='coerce')
        target = pd.Timestamp(time_step)
        frequency = cfg.get('aggregation_frequency')
        if frequency:
            lower = target - pd.Timedelta(frequency)
            selected = (timestamps > lower) & (timestamps <= target)
        else:
            selected = timestamps == target
        return data.loc[selected].copy()

    @classmethod
    def _subset_time(cls, data: pd.DataFrame, cfg: dict, time_step) -> pd.DataFrame:
        """Select and optionally aggregate long records by station code."""
        chosen = cls._select_time(data, cfg, time_step)
        if not cfg.get('aggregation_frequency') or chosen.empty:
            return chosen

        chosen['value'] = pd.to_numeric(chosen['value'], errors='coerce')
        grouped = chosen.groupby('code', as_index=False)['value']
        method = cfg.get('aggregation', 'sum')
        if method == 'sum':
            chosen = grouped.sum(min_count=1)
        elif method == 'mean':
            chosen = grouped.mean()
        else:
            raise ValueError('aggregation must be sum or mean')

        # Aggregation keeps values and codes; recover coordinates from the input table.
        for col in ('lon', 'lat'):
            if col in data:
                locations = data[['code', col]].drop_duplicates('code')
                chosen = chosen.merge(locations, on='code', how='left')
        return chosen

    @classmethod
    def _select_row(cls, data: pd.DataFrame, cfg: dict, time_step) -> pd.Series:
        """Select a wide or per-station record, optionally aggregating values."""
        subset = cls._select_time(data, cfg, time_step)
        if subset.empty:
            return pd.Series(dtype=object)
        if cfg.get('aggregation_frequency'):
            numeric = subset.drop(columns='time').apply(pd.to_numeric, errors='coerce')
            numeric = pd.DataFrame(mask_nodata(numeric.to_numpy(), cfg.get('nodata')),
                                   index=numeric.index, columns=numeric.columns)
            # Filter invalid rainfall before aggregation; all-missing stays NaN.
            numeric = numeric.mask(numeric < 0)
            method = cfg.get('aggregation', 'sum')
            if method == 'sum':
                aggregated = numeric.sum(axis=0, min_count=1)
            elif method == 'mean':
                aggregated = numeric.mean(axis=0)
            else:
                raise ValueError('aggregation must be sum or mean')
            # Keep rainfall numeric when all station values are missing.
            record = aggregated.to_dict()
            record['time'] = pd.Timestamp(time_step)
            return pd.Series(record, dtype=object)
        if len(subset) > 1:
            LOG.warning('More than one matching record; taking last')
        return subset.iloc[-1]

    def read_grid(self, time_step=None, *, source: str = 'grid') -> xr.DataArray:
        """Read a static target grid or optional auxiliary background."""
        cfg = self.config[source]
        path = self._path(cfg['file'], time_step)
        ext = path.suffix.lower()
        if ext in ('.tif','.tiff','.asc'):
            with rasterio.open(path) as src:
                # An explicit SRS declares existing coordinates; it does not reproject.
                srs = cfg.get('srs')
                crs = rasterio.crs.CRS.from_user_input(srs) if srs is not None else src.crs
                if crs is None or crs.to_epsg() != 4326:
                    raise ValueError(f'{path}: raster must use EPSG:4326')
                if src.transform.b != 0 or src.transform.d != 0:
                    raise ValueError('Rotated grids are not supported')
                values = src.read(1, masked=True).astype(float).filled(np.nan)
                values = mask_nodata(values, cfg.get('nodata'), metadata=(() if src.nodata is None else (src.nodata,)))
                lon = src.transform.c + (np.arange(src.width)+.5)*src.transform.a
                lat = src.transform.f + (np.arange(src.height)+.5)*src.transform.e
                da = xr.DataArray(values, dims=('lat','lon'),coords={'lat':lat,'lon':lon},name=cfg.get('variable','precip'))
                da.attrs['crs']='EPSG:4326'
                return da
        if ext in ('.nc','.nc4'):
            with xr.open_dataset(path) as ds:
                var = cfg.get('variable','precip')
                if var not in ds:
                    raise KeyError(f'Variable {var!r} not found in {path}')
                da = ds[var]
                lat_name = cfg.get('lat','lat')
                lon_name = cfg.get('lon','lon')
                da = da.rename({lat_name:'lat',lon_name:'lon'}) if (lat_name!='lat' or lon_name!='lon') else da
                if 'time' in da.dims:
                    if da.sizes['time']==1:
                        da = da.isel(time=0,drop=True)
                    elif time_step is not None:
                        da = da.sel(time=pd.Timestamp(time_step),drop=True)
                    else:
                        raise ValueError('Multitime NetCDF requires a time_step')
                if da.ndim != 2:
                    raise ValueError('NetCDF reference must have 2 spatial dimensions')
                da = da.transpose('lat','lon').load()
                da_values = mask_nodata(da.values, cfg.get('nodata'), metadata=metadata_nodata(da))
                return da.copy(data=da_values)
        raise ValueError(f'Unsupported raster type: {path.suffix}')
