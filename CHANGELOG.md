# Changelog

## 2.0.0 - 2026-10-09

### Refactor

- Organize the installable library under `src/prism/`, with shared handlers
  and utilities in `common/` and the interpolation engine in `griso/`.
- Use `GrisoConfig` and `GrisoInterpolator` as the public interpolation API.
- Add offline fixed- and dynamic-radius examples with shared station/radar inputs.
- Allow an explicit `srs` override for TIFF/ASCII grid and background inputs.

## 0.1.7 - 2026-10-09

- Remove the redundant `libraries/` layer: the GRISO engine now lives at `src/prism/griso/`.
- Update Python imports, workflow, tests and docs to use `prism.griso`.
- Keep the repository-level standalone `workflow/`, and do not include it in the wheel.
- No changes to numerical methods or I/O.

## 0.1.6 - 2026-10-09

- Move the one operational workflow and example JSON files from `src/prism/workflow/` to repository-level `workflow/`.
- Install only `prism.common` and `prism.libraries.griso` in the wheel.
- Update direct script source-tree path, tests, packaging and documentation.
- No scientific or I/O changes.


## 0.1.5 - 2026-10-09

- Adopt standard `src/prism/` Python package layout; move `common/`,
  `libraries/griso/`, and the one `workflow/` inside the namespace.
- Avoid global modules named `common`, `libraries`, or `workflow`.
- Update all internal imports and test paths; preserve numerical algorithm.
- Keep direct script execution and add supported `python -m` usage.
- Include workflow JSON configuration files in installed wheel.



## 0.1.1 - 2026-10-08

- Configurable station/background nodata for the core and operational input reader.
- Defaults to -9999 plus NaN/nonfinite and honors GeoTIFF/CF metadata.
- Sanitize rainfall BEFORE subhourly/hourly aggregation (wide/long/per-station).
- Configurable output raster/NetCDF fill value (default -9999).
- Expose GRISO dynamic correlation thresholds in GrisoConfig.
- Add direct functional API `interpolate_griso(...)`.
- Add targeted nodata and serialization tests.

## 0.1.0 - 2026-10-08

- Experimental standalone package of the GRISO interpolation core.
- Object-oriented public interpolator and isolated numerical functions.
- Input/output/settings/time/logging handlers for workflow execution.
- Fixed and dynamic correlation; CSV timeseries and timestep station inputs.
- GeoTIFF and NetCDF support for grid input/output; basic Arc ASCII support.
- Example case and automated synthetic tests.
