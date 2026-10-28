"""Web: VisionClaw-compatible gateway, opt-in /guide, glasses mode on /frame. Offline, stubbed."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from streetsense.observations import ObservationLog, summarise_frame
from streetsense.types import Alert, Box, Detection, Priority
from streetsense.web import create_app
from streetsense.wonder_agent import GuideResult

TOKEN = "test-gateway-token"


class FakeOut:
    def __init__(self):
        self.t = 1.0
        self.detections = [Detection("car", Box(10, 200, 150, 400), 0.9, track_id=4, attrs={"distance_m": 7.26})]
        self.crosswalk = None
        self.surface = None
        self.spoken = Alert("vehicle:left", "car close on your left", Priority.CRITICAL)


@pytest.fixture
def app(monkeypatch):
    from streetsense import web as webmod

    class StubPipe:
        def __init__(self, *a, **k):
            self.vlm = MagicMock()
            self.vlm.ask_sync.return_value = {
                "summary": "It is safe to cross now. A parked car is on the left.",
                "hazards": [], "crossing_visible": True,
                "pedestrian_signal": "walk", "safe_to_cross": "yes",
            }
            self.observations = ObservationLog()
            self.sam = None
            self.smoother = "votes from older photos"

        def process(self, img, annotate=False):
            return FakeOut()

    monkeypatch.setattr(webmod, "Pipeline", StubPipe)
    monkeypatch.delenv("STREETSENSE_DEMO_TOKEN", raising=False)
    monkeypatch.delenv("STREETSENSE_WONDER", raising=False)
    monkeypatch.setenv("STREETSENSE_GATEWAY_TOKEN", TOKEN)
    a = create_app(no_surface=True)
    a.config["TESTING"] = True
    return a


def _jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((64, 48, 3), 90, np.uint8))
    return buf.tobytes()


def visionclaw_body(question: str) -> dict:
    """Exact shape from the cookbook VisionClaw ChatDTOs tests."""
    uri = "data:image/jpeg;base64," + base64.b64encode(_jpeg_bytes()).decode()
    return {"model": "openclaw/default",
            "messages": [{"role": "user", "content": [{"type": "text", "text": question},
                                                       {"type": "image_url", "image_url": {"url": uri}}]}],
            "user": "visionclaw-test", "stream": False}


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_health_matches_openclaw(app):
    assert app.test_client().get("/health").get_json() == {"ok": True, "status": "live"}


def test_gateway_off_without_token(app, monkeypatch):
    monkeypatch.delenv("STREETSENSE_GATEWAY_TOKEN")
    r = app.test_client().post("/v1/chat/completions", json=visionclaw_body("Can I cross?"), headers=AUTH)
    assert r.status_code == 503


def test_gateway_rejects_bad_token_and_missing_image(app):
    c = app.test_client()
    assert c.post("/v1/chat/completions", json=visionclaw_body("hi"),
                  headers={"Authorization": "Bearer nope"}).status_code == 401
    body = {"model": "openclaw/default", "messages": [{"role": "user", "content": "hi"}]}
    assert c.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 400
    body = visionclaw_body("hi")
    body["stream"] = True
    assert c.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 400


def test_gateway_single_photo_never_says_go(app):
    """Muse says yes, but one photo cannot give a confirmed local walk, so fusion refuses go."""
    r = app.test_client().post("/v1/chat/completions", json=visionclaw_body("Can I cross?"), headers=AUTH)
    assert r.status_code == 200
    j = r.get_json()
    assert j["object"] == "chat.completion"
    text = j["choices"][0]["message"]["content"]
    assert "clear" not in text.lower() and "safe" not in text.lower()
    assert j["streetsense"]["route"] == "crossing" and j["streetsense"]["fuse"]["decision"] != "go"


def test_gateway_free_text_is_sanitized(app):
    r = app.test_client().post("/v1/chat/completions", json=visionclaw_body("What is in front of me?"),
                               headers=AUTH)
    text = r.get_json()["choices"][0]["message"]["content"]
    assert text == "A parked car is on the left."


def test_gateway_rate_limited_per_user(app):
    c = app.test_client()
    assert c.post("/v1/chat/completions", json=visionclaw_body("What is here?"), headers=AUTH).status_code == 200
    assert c.post("/v1/chat/completions", json=visionclaw_body("What is here?"), headers=AUTH).status_code == 429


def test_guide_is_off_by_default(app):
    r = app.test_client().post("/guide", data=_jpeg_bytes())
    assert r.status_code == 404


def test_guide_runs_when_opted_in(app, monkeypatch):
    monkeypatch.setenv("STREETSENSE_WONDER", "1")
    seen = {}

    class FakeGuide:
        def run(self, q, img, tools):
            seen["q"] = q
            seen["current"] = tools.current()
            return GuideResult(speech="Bench on your right.", turns=1)

    app.extensions["streetsense_guide_factory"] = lambda cfg: FakeGuide()
    r = app.test_client().post("/guide?q=what%20is%20near", data=_jpeg_bytes())
    assert r.status_code == 200 and r.get_json()["speech"] == "Bench on your right."
    assert seen["q"] == "what is near"


def test_guide_respects_demo_token(app, monkeypatch):
    monkeypatch.setenv("STREETSENSE_WONDER", "1")
    monkeypatch.setenv("STREETSENSE_DEMO_TOKEN", "abc")
    assert app.test_client().post("/guide", data=_jpeg_bytes()).status_code == 401


def test_frame_glasses_mode_returns_short_cue_with_earcon(app):
    c = app.test_client()
    j = c.post("/frame?mode=glasses", data=_jpeg_bytes()).get_json()
    assert j["glasses"] == {"text": "car close left", "earcon": "alarm", "priority": 3, "key": "vehicle:left"}
    assert "glasses" not in c.post("/frame", data=_jpeg_bytes()).get_json()


def test_page_has_glasses_mode_and_earcons(app):
    c = app.test_client()
    html = c.get("/").get_data(as_text=True)
    assert "__EARCONS__" not in html and '"alarm"' in html and "Glasses mode" in html
    r = c.get("/glasses?res=low&fps=7")
    assert r.status_code == 302 and r.headers["Location"].endswith("/?mode=glasses&res=low&fps=7")


def test_summarise_frame_is_compact_and_json_safe():
    import json

    s = summarise_frame(FakeOut(), 640, 480)
    assert s["objects"][0] == {"label": "car", "zone": "left", "in_path": False, "dist_m": 7.3, "track": 4}
    assert s["signal"] == "unknown" and s["crosswalk"] == {"found": False}
    assert s["spoken"] == "car close on your left"
    json.dumps(s)


def test_gateway_resets_light_votes_per_photo(app):
    from streetsense.traffic_light import LightStateSmoother

    c = app.test_client()
    c.post("/v1/chat/completions", json=visionclaw_body("What is here?"), headers=AUTH)
    sess = app.extensions["streetsense_sessions"]["gw:visionclaw-test"]
    assert isinstance(sess.pipe.smoother, LightStateSmoother)
