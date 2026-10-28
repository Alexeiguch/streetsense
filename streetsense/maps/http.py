"""Tiny JSON-over-HTTP layer with polite rate limits and recorded fixtures.

Fixtures make tests and the demo run offline. Request headers are never stored,
so API keys (sent only in headers) never reach a fixture file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests

USER_AGENT = "StreetSense/0.1 (hackathon walking aid; https://github.com/Alexeiguch/streetsense)"


class FixtureMissing(RuntimeError):
    pass


class Http:
    """mode: live (network only), record (network, then save), replay (fixtures only)."""

    def __init__(self, fixture_dir: str | Path | None = None, mode: str = "live",
                 min_interval_s: float = 1.1, timeout_s: float = 25.0, session: Any = None):
        if mode not in {"live", "record", "replay"}:
            raise ValueError(f"bad http mode {mode}")
        self.mode = mode
        self.dir = Path(fixture_dir) if fixture_dir else None
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.session = session or requests.Session()
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()
        self.calls: list[dict] = []

    @staticmethod
    def key(method: str, url: str, body: Any) -> str:
        blob = json.dumps([method, url, body], sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _path(self, method: str, url: str, body: Any) -> Path:
        host = (urlparse(url).hostname or "x").split(".")[0]
        assert self.dir is not None
        return self.dir / f"{host}_{self.key(method, url, body)}.json"

    def _throttle(self, host: str) -> None:
        with self._lock:
            wait = self._last.get(host, 0.0) + self.min_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.monotonic()

    def request(self, method: str, url: str, *, params: dict | None = None, json_body: Any = None,
                form: dict | None = None, headers: dict | None = None) -> Any:
        full = url + ("?" + urlencode(sorted(params.items())) if params else "")
        body = json_body if json_body is not None else form
        if self.mode == "replay" or (self.mode == "record" and self.dir and self._path(method, full, body).exists()):
            if not self.dir:
                raise FixtureMissing("replay mode needs a fixture_dir")
            p = self._path(method, full, body)
            if not p.exists():
                raise FixtureMissing(f"no fixture for {method} {urlparse(full).hostname}{urlparse(full).path}")
            self.calls.append({"method": method, "host": urlparse(full).hostname, "fixture": p.name})
            return json.loads(p.read_text())["response"]
        host = urlparse(full).hostname or ""
        self._throttle(host)
        h = {"User-Agent": USER_AGENT, **(headers or {})}
        t0 = time.perf_counter()
        r = self.session.request(method, full, json=json_body, data=form, headers=h, timeout=self.timeout_s)
        ms = round((time.perf_counter() - t0) * 1000)
        self.calls.append({"method": method, "host": host, "status": r.status_code, "ms": ms})
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        data = _parse_body(r.text, ctype)
        if self.mode == "record" and self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            p = self._path(method, full, body)
            redacted_url = full  # keys travel in headers, never in the URL
            p.write_text(json.dumps({"request": {"method": method, "url": redacted_url, "body": body},
                                     "response": data}, indent=1, default=str))
        return data


def _parse_body(text: str, ctype: str) -> Any:
    if "text/event-stream" in ctype:
        # MCP Streamable HTTP may answer as SSE; keep the last JSON data event.
        last = None
        for line in text.splitlines():
            if line.startswith("data:"):
                try:
                    last = json.loads(line[5:].strip())
                except ValueError:
                    pass
        return last
    return json.loads(text) if text.strip() else None


def default_fixture_dir() -> Path:
    return Path(os.environ.get("STREETSENSE_MAPS_FIXTURES", Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "maps"))
