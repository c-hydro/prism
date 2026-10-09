# PRISM GRISO 2.0.0

PRISM is a Python library for interpolating rain-gauge observations onto a
regular geographic grid with GRISO. It supports fixed correlation kernels
and dynamic kernels estimated from an auxiliary gridded rainfall field.

The package provides an in-memory Python API and a configurable workflow
for reading station data, processing timesteps and writing rainfall grids.

Developed by CIMA Research Foundation.
Authors: Andrea Libertino and Flavio Pignone.

## Project layout

```text
prism/                           # Repository root
  src/
    prism/                       # Installable Python package
      __init__.py
      common/
        input_handler.py
        output_handler.py
        settings_handler.py
        time_utils.py
        logging_handler.py
        data_utils.py
        geo_utils.py
      griso/
        griso_interpolator.py
        griso_preprocessing.py
        griso_correlation.py
  workflow/                      # Workflow and JSON settings
    prism_griso.py               # Single workflow script
    prism_griso_settings_fixed.json
    prism_griso_settings_dynamic.json
  examples/
    example/
      raingauge_cell_value_20261008_1200.txt  # Station observations
      SRT1_202610081200.tif                  # Geographic grid and radar rainfall
  CHANGELOG.md
  LICENSE
  pyproject.toml
```

`prism.common` contains input, output and settings handlers, together with
shared data, time and spatial utilities. `prism.griso` separates grid and
station preprocessing, correlation-kernel construction and interpolation.

The wheel installs the `prism` package. The workflow, JSON configuration
files and example data are available in the repository checkout.

## Install

Python >= 3.10:

```bash
python -m pip install -e .
```

For NetCDF support with the optional netCDF4 backend:

```bash
python -m pip install -e '.[netcdf]'
```

The examples share saved station observations and a radar raster for
`2026-10-08 12:00`. Both run offline, without server access or station downloads.
The station file has no header and contains longitude, latitude and rainfall
in three whitespace-separated columns. The observations are provided as saved,
including a maximum value of `2268.0012`; they are not a quality-controlled dataset.

## Run the workflow

The workflow entrypoint is `workflow/prism_griso.py`.
From the repository root, after `python -m pip install -e .` (or directly
from the checkout if runtime dependencies are available):

```bash
python workflow/prism_griso.py \
    -settings_file workflow/prism_griso_settings_fixed.json \
    -time '2026-10-08 12:00' \
    -domain example
```

For dynamic correlation:

```bash
python workflow/prism_griso.py \
    -settings_file workflow/prism_griso_settings_dynamic.json \
    -time '2026-10-08 12:00' \
    -domain example
```

The fixed example uses a 30 km radius and the radar only for grid geometry.
The dynamic example estimates local radii between 5 and 30 km from radar rainfall.
Both interpolate the same station observations and write GeoTIFFs:
`examples/example/output/griso_fixed_202610081200.tif` and
`examples/example/output/griso_dynamic_202610081200.tif`.
Existing output files are skipped unless `flags.overwrite` is `true`.

Only this timestep is included. To process other periods, provide their input
files and use `time.steps` or `-start` and `-end`. Generated outputs and logs
are excluded from Git.

## Workflow settings

The JSON configuration contains:

- `settings.domain`: domain tag used in file paths; `-domain` overrides it.
- `time`: timestep frequency, number of steps and direction (`backward` or
  `forward`). `-start` and `-end` select an explicit inclusive range.
- `algorithm`: `GrisoConfig` parameters, including `radius_km`,
  `min_radius_km` and `correlation` (`fixed` or `dynamic`).
- `input`: station and target-grid settings, plus `background` for dynamic mode.
- `outcome`: output path and optional nodata value.
- `log`: optional log path.
- `flags`: `debug`, `overwrite` and `skip_missing_steps`.

Paths support `{domain}` and date tokens such as `%Y%m%d%H%M`. Per-station
paths also accept `{code}` and `{station_name}`. Relative paths are resolved
from the working directory, so run the example commands from the repository root.

`skip_missing_steps` allows processing to continue when a timestep encounters
an input, validation or interpolation error. Successfully processed timesteps
are written to the configured output path.

## Use GRISO directly from Python

```python
import pandas as pd
import xarray as xr
from prism.griso import GrisoInterpolator, GrisoConfig

# Grid: geographic lat/lon raster with 1D coordinates (EPSG:4326)
grid = xr.DataArray(
    [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    dims=('lat', 'lon'), coords={'lat':[44.0,44.01,44.02],
                                'lon':[8.0,8.01,8.02]})

stations = pd.DataFrame({
    'code':['S1','S2'], 'lon':[8.005,8.015],
    'lat':[44.005,44.015], 'value':[4.0,8.0]})

interp = GrisoInterpolator(GrisoConfig(radius_km=30,correlation='fixed'))
rain = interp.interpolate(stations,grid) # xarray.DataArray
```

For dynamic correlation, instantiate with `correlation='dynamic'`, then supply
`background=<DataArray>` with the **same** lat/lon coordinates as the grid.
The background determines local correlation lengths. The resulting rainfall
field is a weighted combination of the station observations.

## Station inputs

`InputHandler` returns a `code, lon, lat, value` table for every supported
station input layout. The interpolation API requires `lon`, `lat` and `value`;
station codes are optional for direct Python calls.

Supported `input.stations.type`:

1. `timestep`: a CSV per timestep with station coordinates and rainfall; or
   a station-values CSV with code plus an external coordinate CSV.
2. `timeseries`, `layout: wide`: one CSV with `time,S1,S2,...`, joined to
   `coordinates.file` containing `code,lat,lon`.
3. `timeseries`, `layout: long`: one CSV with `time,code,value`, with an
   external coordinate CSV, or `time,code,lon,lat,value` without one.
4. `timeseries`, `layout: per_station`: one file per station (columns
   `time,value`) and separate coordinates; template can use `{code}`.

Column names can be set with a `columns` mapping from standard names to
source column labels or zero-based positions. For headerless CSV set
`header: false`. The default is an exact match to the requested timestamp.
For subhourly data, configure `aggregation_frequency: '1h'` and
`aggregation: 'sum'` (or `'mean'`). Aggregation uses `(t - 1h, t]`.
A series with zero available observations is **not** treated as zero rainfall.

Example wide series:

```json
"stations": {
  "type": "timeseries", "layout": "wide",
  "file": "/data/rain.csv",
  "coordinates": {"file": "/data/stations.csv"}
}
```

Example per-station series:

```json
"stations": {
  "type": "timeseries", "layout": "per_station",
  "file": "/data/stations/{code}.csv",
  "coordinates": {"file": "/data/stations_coords.csv"},
  "aggregation_frequency": "1h", "aggregation": "sum"
}
```


## Configurable missing-data policy

**Defaults** (without any extra JSON or Python arguments):

- **Station records:** `-9999`, `NaN`, `Inf` and negative rainfall values are
  rejected. **Zero is valid rainfall.** A completely missing aggregation window
  does not become zero.
- **Raster background and grid:** `-9999`, `NaN`, `Inf` are masked; GeoTIFF
  `nodata` metadata and NetCDF/CF `_FillValue` or `missing_value` are also
  honored automatically on **each input file/timestep**.
- **Output raster:** missing cells are encoded as `-9999` by default.

A `nodata` setting may be **one number or an array**. When omitted it defaults
to `[-9999]`; use `[]` to disable that default sentinel while still honoring
NaN/nonfinite and input-file metadata. Source-specific metadata is applied
in addition to the configured values. For instance:

```json
{
  "input": {
    "stations": {"type": "timeseries", "nodata": [-9999, -8888]},
    "background": {"file": "/data/radar_%Y%m%d%H%M.tif", "nodata": -32768}
  },
  "outcome": {"file": "/data/griso.tif", "nodata": -9999}
}
```

The other required workflow fields (e.g. input file and grid definition)
are omitted from this illustrative snippet.

**Nodata is masked before temporal aggregation**, for all station time-series
layouts, not after the accumulated value has been computed. If a timestep
contains only missing station observations, GRISO raises a clear error; the
workflow may skip that timestep using `flags.skip_missing_steps`.

For Python callers, the core is independent of the reader and can normalize
custom nodata values itself:

```python
from prism.griso import GrisoConfig, GrisoInterpolator

# Example auxiliary field on the target grid; replace with your rainfall data:
radar = xr.ones_like(grid)

# Reuse a configurable class across many timesteps:
engine = GrisoInterpolator(GrisoConfig(
    correlation='dynamic', radius_km=30,
    station_nodata=(-9999, -8888), background_nodata=(-32768,)))
rain = engine.interpolate(stations, grid, background=radar)

# Override missing-data settings for an individual call:
rain = engine.interpolate(stations, grid, background=radar, station_nodata=[-8888])
```

The direct `interpolate()` method accepts per-call `station_nodata` and
`background_nodata` overrides. Thresholds associated with dynamic fitting
are available through `GrisoConfig`: `dynamic_min_valid_background`,
`dynamic_wet_gauge_threshold`, `dynamic_valid_pixels_factor`, and
`kernel_cutoff`. Their defaults are `0.0`, `0.2`, `100.0` and `0.001`,
respectively. The default maximum and minimum correlation radii are
`30.0 km` and `5.0 km`.

## Grids and outputs

- Target grid: **GeoTIFF**, ESRI ASCII (`.asc`) or NetCDF (`.nc` / `.nc4`).
  TIFF/ASCII must declare EPSG:4326 in their CRS metadata or through `srs`.
  Geographical coordinates only,
  with one-dimensional longitude and latitude coordinates and regular spacing.
- Optional dynamic-correlation background: same formats and *identical*
  target grid coordinates.
- Output: GeoTIFF (`.tif`), NetCDF (`.nc`) or ESRI ASCII (`.asc`).
- Output is always nonnegative rainfall; negative/missing gauge readings are
  discarded, colocated gauge values are averaged, out-of-domain gauges ignored.
- GeoTIFF output automatically uses north-up orientation, independently of the
  grid input latitude ordering. Units are carried only when supplied by the
  grid. No automatic unit conversion is performed.

For an ASCII grid without CRS metadata, declare its coordinate system explicitly:

```json
"grid": {
  "file": "/data/grid.asc",
  "srs": "EPSG:4326"
}
```

For TIFF/ASCII inputs, `srs` overrides the file CRS, even when the file already
declares one. If omitted or `null`, the file CRS is used. Only EPSG:4326 is
supported: this setting does not reproject coordinates or modify the input file.
The coordinates must already be WGS84 longitude/latitude in degrees.
Dynamic `input.background` accepts its own `srs` setting; it does not inherit
the target grid override.

## Interpolation and grid requirements

Stations are mapped to their nearest grid cells using a KD-tree on
unit-sphere coordinates. Observations sharing a cell are averaged before
the station correlation system is assembled and solved by least squares.
Weighted spherical kernels are then summed on the target grid.

The grid must be two-dimensional, with monotonic, regularly spaced `lat`
and `lon` coordinates. Projected, rotated and curvilinear grids are not
supported. Dynamic mode requires a background on exactly the same grid;
the workflow does not reproject or resample inputs.

Kernel distances use an approximate isotropic kilometre step derived from
latitude spacing. The radius is rounded to an even number of cells and the
effective step is adjusted to match the requested radius. This approximation
should be considered when choosing grid resolution and geographic extent.

Station values and the background must use consistent rainfall units and
accumulation periods. Temporal aggregation supports sums and means;
unit conversion is the caller's responsibility.

## License

European Union Public Licence v1.2 (EUPL-1.2).
The complete text is available in [LICENSE](LICENSE).
