"""Opt-in walking navigation for StreetSense.

OpenStreetMap (Valhalla routing in blind mode, Overpass crossing tags, Nominatim
places) drives the spoken turn-by-turn and crossing previews. Google Maps is used
only through Maps Experience agent tools for Wonder place questions, because the
Google Maps Platform Terms forbid text to speech of raw Routes or Places content.

Map data never authorises a crossing. The go rule stays: local reader says walk
AND a validated Muse Spark reply says yes (streetsense/answers.py).
"""

from __future__ import annotations

from typing import Any


def build_session(cfg: Any = None) -> Any:
    """MapSession from a MapsConfig (or the environment)."""
    from ..config import MapsConfig
    from .google_agent import GoogleMapsAgentTools
    from .http import Http, default_fixture_dir
    from .osm import OSMProvider
    from .session import MapSession

    cfg = cfg or MapsConfig.from_env()
    fx = cfg.fixtures_dir or default_fixture_dir()
    if cfg.provider == "fixture":
        http = Http(fx, mode="replay")
    elif cfg.provider == "record":
        http = Http(fx, mode="record", timeout_s=60)
    elif cfg.provider == "osm":
        http = Http(timeout_s=30)
    else:
        raise ValueError(f"unknown maps provider {cfg.provider!r}; use osm, fixture or record")
    google = GoogleMapsAgentTools() if cfg.google and cfg.google_key_present else None
    return MapSession(OSMProvider(http), google=google)
