"""
Script Features:
GRISO rainfall interpolation from station observations to a geographic grid.
Fixed or dynamic correlation with configurable inputs, outputs and timesteps.

Name:          prism_griso
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'

General command line:
python workflow/prism_griso.py -settings_file workflow/prism_griso_settings_fixed.json -time '2026-10-08 12:00' -domain example
or
python workflow/prism_griso.py -settings_file workflow/prism_griso_settings_fixed.json -start '2026-10-08 12:00' -end '2026-10-08 16:00' -domain example
"""
# Import libraries
import argparse
import logging
import sys
from pathlib import Path

# Load the source package when this script is run directly from the repository.
if __package__ in (None, ""):
    _src = Path(__file__).resolve().parents[1] / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from prism.griso import GrisoInterpolator
from prism.common import InputHandler, OutputHandler, SettingsHandler
from prism.common.time_utils import time_steps, resolve_path
from prism.common.logging_handler import setup_logging

LOG = logging.getLogger(__name__)


def run(settings_file: str | Path, reference: str | None = None,
        domain: str | None = None, start: str | None = None,
        end: str | None = None) -> list[Path]:
    """Read station data, interpolate rainfall and write each requested timestep."""
    # Load settings and resolve the domain and processing period.
    config = SettingsHandler(settings_file, domain).get()
    name = config['settings'].get('domain')
    flags = config.get('flags', {})
    periods = time_steps(reference, config.get('time', {}), start, end)
    
    # Setup logging
    logfile = config.get('log', {}).get('file')
    setup_logging(resolve_path(logfile, periods[-1], name) if logfile else None,
                  debug=flags.get('debug', False))

    # Initialize readers, writer and interpolation parameters from settings.
    reader = InputHandler(config['input'], domain=name)
    writer = OutputHandler(config['outcome'], domain=name)
    griso = GrisoInterpolator(config.get('algorithm', {}))

    # Read the reference grid 
    target_grid = reader.read_grid(periods[0])
    results = []

    # Process each timestep 
    for time_step in periods:
        out = writer.path(time_step)
        if out.exists() and not flags.get('overwrite', False):
            LOG.info('Output exists, skipping: %s', out)
            continue
        try:
            LOG.info('Processing %s', time_step)
            stations = reader.read_stations(time_step)
            # If dynamic correlation is used, read the background field.
            background = (reader.read_grid(time_step, source='background')
                          if griso.config.correlation == 'dynamic' else None)
            # Perform the interpolation using the GRISO algorithm
            rainfall = griso.interpolate(
                stations, target_grid, background=background,
                station_nodata=config['input']['stations'].get('nodata'),
                background_nodata=config['input'].get('background', {}).get('nodata'),
            )
        except (OSError, ValueError, KeyError) as exc:
            # Skipping failed timesteps is controlled by the workflow settings.
            if flags.get('skip_missing_steps', False):
                LOG.warning('Skipping time step %s: %s', time_step, exc)
                continue
            raise
        path = writer.write(rainfall, time_step, overwrite=flags.get('overwrite', False))
        LOG.info('Saved %s', path)
        results.append(path)
    return results


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and return the workflow exit status."""
    parser = argparse.ArgumentParser(description='GRISO interpolator workflow')
    parser.add_argument('-settings_file', required=True)
    parser.add_argument('-time', '--time', default=None, help='Reference time YYYY-MM-DD HH:MM')
    parser.add_argument('-start', '--start', default=None, help='First timestep (optional)')
    parser.add_argument('-end', '--end', default=None, help='Last timestep (optional)')
    parser.add_argument('-domain', '--domain', default=None)
    args = parser.parse_args(argv)

    try:
        results = run(args.settings_file, args.time, args.domain, args.start, args.end)
    except Exception:
        LOG.exception('GRISO workflow failed')
        return 1
    print(f'Processed {len(results)} time step(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
