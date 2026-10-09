"""
Library Features:
Utilities for managing time, generating time steps, and handling time-dependent paths

Name:          time_utils
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""
from datetime import datetime
import pandas as pd

TIME_FORMAT = '%Y-%m-%d %H:%M'


def parse_time(value: str | datetime) -> pd.Timestamp:
    """Parse a timestamp or a string in the configured command-line format."""
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value)
    try:
        return pd.Timestamp(datetime.strptime(value, TIME_FORMAT))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Time must be '{TIME_FORMAT}', got {value!r}") from exc


def time_steps(reference: str | None, settings: dict,
               start: str | None = None, end: str | None = None) -> list[pd.Timestamp]:
    """Build timesteps from an explicit range or a reference time and step count."""
    freq = settings.get('frequency', '1h')
    if start is not None or end is not None:
        if not start or not end:
            raise ValueError('Provide both -start and -end')
        rng = pd.date_range(parse_time(start), parse_time(end), freq=freq)
    else:
        if reference is None:
            raise ValueError('Provide -time or both -start and -end')
        count = int(settings.get('steps', 1))
        if count < 1:
            raise ValueError('time.steps must be >= 1')
        ref = parse_time(reference)
        if settings.get('direction', 'backward') == 'backward':
            rng = pd.date_range(end=ref, periods=count, freq=freq)
        elif settings.get('direction') == 'forward':
            rng = pd.date_range(start=ref, periods=count, freq=freq)
        else:
            raise ValueError('time.direction must be backward or forward')
    return list(rng)


def resolve_path(path: str, time_step: pd.Timestamp | None = None,
                 domain: str | None = None, **kwargs: str) -> str:
    """Replace named tags, then format date tokens using the timestep."""
    output = str(path)
    for name, value in {'domain': domain, **kwargs}.items():
        if value is not None:
            output = output.replace('{' + name + '}', str(value))
    if time_step is not None:
        output = pd.Timestamp(time_step).strftime(output)
    return output
