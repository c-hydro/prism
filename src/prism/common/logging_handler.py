"""
Library Features:
Utilities for setting up and managing logging

Name:          logging_handler
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""
import logging
from pathlib import Path


def setup_logging(filename: str | None = None, debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if filename:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))
    logging.basicConfig(level=level, handlers=handlers,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        force=True)
