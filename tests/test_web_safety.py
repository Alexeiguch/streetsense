"""Web boundary: empty /frame → 400, per-session isolation, /ask auth & single-flight."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from streetsense.types import Alert, Box, Detection, Priority
from streetsense.web import create_app


class FakeOut:
    def __init__(self):
        self.detections = [
            Detection("person", Box(500, 200, 700, 700), 0.9, attrs={"distance_m": 2.0}),
        ]
        self.crosswalk = None
        self.surface = None
        self.spoken = Alert("person:ahead", "person ahead, 2 metres", Priority.WARNING)


@pytest.fixture
def app(monkeypatch):
    # Avoid loading real YOLO weights: replace Pipeline with a stub.
    from streetsense import web as webmod

    class StubPipe:
        def __init__(self, *a, **k):
            self.vlm = MagicMock()
            self.vlm.ask_sync.return_value = {
                "summary": "wait",
                "hazards": [],
                "crossing_visible": True,
                "pedestrian_signal": "red",
                "safe_to_cross": "no",
            }
            self.alerts = MagicMock()
            self._n = 0

        def process(self, img, annotate=False):
            self._n += 1
            return FakeOut()

    monkeypatch.setattr(webmod, "Pipeline", StubPipe)
    application = create_app(no_surface=True)
    application.config["TESTING"] = True
    return application


def _jpeg() -> bytes:
    import cv2
    img = np.full((48, 64, 3), 40, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_empty_frame_returns_400(app):
    c = app.test_client()
    r = c.post("/frame", data=b"")
    assert r.status_code == 400
    assert "empty" in r.get_json()["error"]


def test_garbage_frame_returns_400(app):
    c = app.test_client()
    r = c.post("/frame", data=b"not-a-jpeg")
    assert r.status_code == 400


def test_sessions_are_isolated(app):
    """Two clients get independent Pipeline state (cooldowns / trackers)."""
    c1 = app.test_client()
    c2 = app.test_client()
    jpg = _jpeg()
    r1 = c1.post("/frame", data=jpg)
    r2 = c2.post("/frame", data=jpg)
    assert r1.status_code == 200 and r2.status_code == 200
    s1 = r1.headers.getlist("Set-Cookie")
    s2 = r2.headers.getlist("Set-Cookie")
    # Different session cookies
    assert any("ss_sid=" in x for x in s1)
    assert any("ss_sid=" in x for x in s2)
    sessions = app.extensions["streetsense_sessions"]
    assert len(sessions) >= 2


def test_ask_requires_token_when_configured(app, monkeypatch):
    monkeypatch.setenv("STREETSENSE_DEMO_TOKEN", "secret-demo-token")
    c = app.test_client()
    r = c.post("/ask", data=_jpeg())
    assert r.status_code == 401
    r = c.post("/ask", data=_jpeg(), headers={"X-Demo-Token": "wrong"})
    assert r.status_code == 401
    r = c.post("/ask", data=_jpeg(), headers={"X-Demo-Token": "secret-demo-token"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["spoken"]
    assert "cross now" not in body["spoken"].lower() or "can't" in body["spoken"].lower() or "do not" in body["spoken"].lower()


def test_ask_single_flight(app, monkeypatch):
    monkeypatch.delenv("STREETSENSE_DEMO_TOKEN", raising=False)
    c = app.test_client()
    # Mark session inflight manually after first cookie
    r = c.post("/frame", data=_jpeg())
    assert r.status_code == 200
    sessions = app.extensions["streetsense_sessions"]
    assert sessions
    sid = next(iter(sessions))
    sessions[sid].ask_inflight = True
    r2 = c.post("/ask", data=_jpeg())
    assert r2.status_code == 409
