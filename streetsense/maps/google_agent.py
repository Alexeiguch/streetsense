"""Google Maps via Maps Experience agent tools (formerly Maps Grounding Lite), the MCP server
Google provides for grounding third-party LLMs.

Why this and not the raw Routes or Places REST APIs: the Google Maps Platform
Terms (3.2.3) forbid using Google Maps Content with text-to-speech and forbid
turning directions into synthesized speech. StreetSense is a voice app, so raw
Routes steps or Places names cannot be spoken. The agent tools come with an
explicit path for voice-first apps: show the Google Maps sources in a
companion UI with a verbatim transcript, tell the user that answers may
include Google Maps content, and use an LLM that does not train on inputs
(Muse Wonder on Model API Standard mode does not).

What we use: search_places (places, opening hours, accessibility questions)
and compute_routes (WALK distance and duration only; it has no turn-by-turn).
Turn-by-turn and crossing details stay on OpenStreetMap. We do not feed Google
coordinates into OSM routing, to respect the no non-Google map rule.

The key is read from GOOGLE_MAPS_API_KEY, sent only in the X-Goog-Api-Key
header, and never logged or written to fixtures.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

MCP_URL = os.environ.get("STREETSENSE_GOOGLE_MCP_URL", "https://mapstools.googleapis.com/mcp")
DISCLOSURE = "Some of this comes from Google Maps. Sources are in the app."
PERSISTENT_DISCLOSURE = (
    "Answers from the guide may include Google Maps content. Each answer that uses it lists its "
    "Google Maps sources in the transcript on this page."
)


@dataclass
class GoogleSource:
    title: str
    url: str

    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "attribution": "Google Maps"}


@dataclass
class GoogleToolResult:
    tool: str
    data: dict
    sources: list[GoogleSource] = field(default_factory=list)


def _mask(text: str, key: str | None) -> str:
    if key:
        text = text.replace(key, "***")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", text)


class GoogleMapsAgentTools:
    """Minimal MCP client (JSON-RPC over Streamable HTTP) for the two tools we need."""

    def __init__(self, api_key: str | None = None, url: str = MCP_URL, post: Any = None, timeout_s: float = 20.0):
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_MAPS_API_KEY")
        self.url = url
        self.timeout_s = timeout_s
        self._post = post  # injectable for tests: post(payload, headers) -> (json, headers)
        self._session_id: str | None = None
        self._initialized = False
        self._id = 0
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _send(self, method: str, params: dict | None = None, notify: bool = False) -> Any:
        payload: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            self._id += 1
            payload["id"] = self._id
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "X-Goog-Api-Key": self.api_key or ""}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self._post is not None:
            body, resp_headers = self._post(payload, headers)
        else:
            import requests

            from .http import USER_AGENT, _parse_body

            r = requests.post(self.url, json=payload, headers={"User-Agent": USER_AGENT, **headers}, timeout=self.timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(_mask(f"Google MCP HTTP {r.status_code}: {r.text[:200]}", self.api_key))
            body, resp_headers = _parse_body(r.text, r.headers.get("content-type", "")), dict(r.headers)
        sid = {k.lower(): v for k, v in (resp_headers or {}).items()}.get("mcp-session-id")
        if sid:
            self._session_id = sid
        if notify:
            return None
        if isinstance(body, dict) and body.get("error"):
            raise RuntimeError(_mask(f"Google MCP error: {str(body['error'])[:200]}", self.api_key))
        return (body or {}).get("result")

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                  "clientInfo": {"name": "streetsense", "version": "0.1"}})
        self._send("notifications/initialized", notify=True)
        self._initialized = True

    def call(self, tool: str, arguments: dict) -> GoogleToolResult:
        if not self.enabled:
            raise RuntimeError("Google Maps disabled: set GOOGLE_MAPS_API_KEY")
        self._ensure_init()
        self.calls += 1
        result = self._send("tools/call", {"name": tool, "arguments": arguments}) or {}
        data = result.get("structuredContent")
        if data is None:
            text = "".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
            try:
                data = json.loads(text)
            except ValueError:
                data = {"text": text}
        return GoogleToolResult(tool=tool, data=data if isinstance(data, dict) else {"result": data},
                                sources=extract_sources(data))

    def search_places(self, text_query: str, near: tuple[float, float] | None = None) -> GoogleToolResult:
        args: dict = {"text_query": text_query[:200]}
        if near:
            args["location_bias"] = {"circle": {"center": {"latitude": near[0], "longitude": near[1]}, "radius_meters": 1500}}
        return self.call("search_places", args)

    def walk_time(self, origin: tuple[float, float], destination: str) -> GoogleToolResult:
        dest = {"place_id": destination} if destination.startswith("ChIJ") else {"address": destination[:200]}
        return self.call("compute_routes", {"origin": {"lat_lng": {"latitude": origin[0], "longitude": origin[1]}},
                                            "destination": dest, "travel_mode": "WALK"})


def extract_sources(data: Any) -> list[GoogleSource]:
    """Collect Google Maps links from a tool result, wherever the server puts them."""
    out: list[GoogleSource] = []
    seen: set[str] = set()

    def walk(x: Any, title: str = "") -> None:
        if isinstance(x, dict):
            t = x.get("displayName") or x.get("name") or x.get("title") or title
            if isinstance(t, dict):
                t = t.get("text", "")
            links = x.get("googleMapsLinks")
            if isinstance(links, dict):
                for k in ("placeUrl", "directionsUrl", "reviewsUrl"):
                    u = links.get(k)
                    if u and u not in seen:
                        seen.add(u)
                        out.append(GoogleSource(title=str(t or "Google Maps"), url=u))
                        break
            for k in ("googleMapsUri", "uri", "url"):
                u = x.get(k)
                if isinstance(u, str) and "google." in u and u not in seen:
                    seen.add(u)
                    out.append(GoogleSource(title=str(t or "Google Maps"), url=u))
            for v in x.values():
                walk(v, str(t or ""))
        elif isinstance(x, list):
            for v in x:
                walk(v, title)

    walk(data)
    return out
