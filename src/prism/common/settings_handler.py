"""
Library Features:
Utilities for managing JSON settings files

Name:          settings_handler
Author(s):     Andrea Libertino (andrea.libertino@cimafoundation.org)
               Flavio Pignone (flavio.pignone@cimafoundation.org)
Date:          '20261009'
Version:       '2.0.0'
"""
from copy import deepcopy
import json
from pathlib import Path


class SettingsHandler:
    def __init__(self, settings_file: str | Path, domain: str | None = None):
        self.path = Path(settings_file)
        with self.path.open(encoding='utf-8') as stream:
            self.settings = json.load(stream)
        if domain is not None:
            self.settings.setdefault('settings', {})['domain'] = domain
        if not {'settings', 'input', 'outcome'}.issubset(self.settings):
            raise ValueError("Settings JSON needs 'settings', 'input' and 'outcome'")
        self.domain = self.settings['settings'].get('domain')

    def get(self) -> dict:
        """Return a copy so callers can edit settings independently."""
        return deepcopy(self.settings)
