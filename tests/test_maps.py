"""Maps navigation: offline (recorded OSM fixtures, mocked Google). Map data never authorises crossing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from streetsense import meta_usage
from streetsense.config import MapsConfig, WonderConfig
from streetsense.maps import build_session, geo
from streetsense.maps.google_agent import DISCLOSURE, GoogleMapsAgentTools, extract_sources
from streetsense.maps.http import FixtureMissing, Http, default_fixture_dir
from streetsense.maps.model import Crossing, Route, Step, choose_safer_route
from streetsense.maps.navigator import Fix, NavConfig, Navigator
from streetsense.maps.osm import OSMProvider, attach_crossings, crossing_from_tags, crossings_from_overpass
from streetsense.maps.walk_sim import TrackConfig, run_walk, synth_track, to_srt
from streetsense.wonder_agent import _FORBIDDEN, GuideTools, WonderGuide

FROM, TO = (38.9031, -77.0387), (38.8992, -77.0388)
FAKE_KEY = "AIzaFAKEKEYFORTESTS0000000000000000000"


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("STREETSENSE_META_BUDGET_USD", "1.0")
    led = meta_usage.UsageLedger(tmp_path / "usage.jsonl")
    monkeypatch.setattr(meta_usage, "_LEDGER", led)
    return led


@pytest.fixture(scope="module")
def dc_routes():
    return OSMProvider(Http(default_fixture_dir(), mode="replay")).routes(FROM, TO)


def straight_route(n_m=200.0, crossings=(), steps=None):
    pts = [geo.destination(FROM, 180.0, d) for d in range(0, int(n_m) + 1, 10)]
    r = Route(points=pts, steps=steps or [Step("Walk south.", n_m, 0, "depart")], distance_m=n_m, duration_s=n_m / 1.3)
    for along in crossings:
        p = geo.destination(FROM, 180.0, along)
        r.crossings.append(Crossing(p[0], p[1], kind="traffic_signals", signals=True, road="K Street", along_m=along))
    return r


def permits_crossing(text: str) -> bool:
    return any(p.search(text) for p in _FORBIDDEN)


# geo ------------------------------------------------------------------------

def test_geo_distance_bearing_clock_and_polyline():
    p = geo.destination(FROM, 90.0, 100.0)
    assert geo.haversine_m(FROM, p) == pytest.approx(100.0, abs=0.5)
    assert geo.bearing_deg(FROM, p) == pytest.approx(90.0, abs=0.5)
    assert geo.clock_face(0) == 12 and geo.clock_face(90) == 3 and geo.clock_face(-90) == 9
    assert geo.relative_bearing(10, 350) == pytest.approx(20)
    line = [FROM, p, TO]
    for prec in (5, 6):
        back = geo.decode_polyline(geo.encode_polyline(line, prec), prec)
        assert all(abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5 for a, b in zip(line, back))
    assert geo.round_m(23) == 25 and geo.round_m(87) == 90


def test_project_onto_line_reports_lateral_offset():
    r = straight_route()
    off = geo.destination(geo.destination(FROM, 180, 50), 90, 15)
    lateral, along, _, _ = geo.project_onto_line(off, r.points, r.cum)
    assert lateral == pytest.approx(15, abs=1) and along == pytest.approx(50, abs=1)


# OSM parsing and safer route --------------------------------------------------

def test_crossing_tags_map_to_risk_classes():
    c = crossing_from_tags(1, 2, {"crossing": "traffic_signals", "traffic_signals:sound": "yes", "tactile_paving": "yes"})
    assert c.signals and c.sound and c.tactile and c.risk_class == "signals_accessible"
    assert crossing_from_tags(1, 2, {"crossing": "uncontrolled", "crossing:markings": "zebra"}).risk_class in {"zebra", "marked"}
    assert crossing_from_tags(1, 2, {"crossing": "unmarked"}).risk_class == "unmarked"
    assert crossing_from_tags(1, 2, {}).risk_class == "unknown"


def test_overpass_parsing_takes_road_name_and_lanes_from_parent_way():
    data = {"elements": [
        {"type": "node", "id": 1, "lat": 38.9, "lon": -77.0, "tags": {"highway": "crossing", "crossing": "marked"}},
        {"type": "way", "id": 9, "nodes": [1, 2], "tags": {"highway": "secondary", "name": "K Street", "lanes": "4"}},
    ]}
    (c,) = crossings_from_overpass(data)
    assert c.road == "K Street" and c.lanes == 4 and c.osm_id == 1


def test_attach_crossings_keeps_only_points_on_the_route():
    r = straight_route()
    near = geo.destination(geo.destination(FROM, 180, 80), 90, 5)
    far = geo.destination(geo.destination(FROM, 180, 80), 90, 60)
    kept = attach_crossings(r, [Crossing(near[0], near[1], kind="zebra"), Crossing(far[0], far[1], kind="zebra")])
    assert len(kept) == 1 and kept[0].along_m == pytest.approx(80, abs=2)


def test_recorded_dc_routes_and_safer_choice(dc_routes):
    assert len(dc_routes) == 3
    best, why = choose_safer_route(dc_routes)
    assert best is not dc_routes[0]
    assert len(best.crossings) == 4 and all(c.signals for c in best.crossings)
    assert best.stairs == 0
    assert "without signals" in why and "shorter" in why
    assert best.speakable and "OpenStreetMap" in best.attribution


def test_safer_choice_respects_max_detour():
    short_risky = straight_route(200, crossings=())
    short_risky.crossings.append(Crossing(0, 0, kind="unmarked", along_m=50))
    long_safe = straight_route(250)
    assert choose_safer_route([short_risky, long_safe], max_detour=1.1)[0] is short_risky
    best, why = choose_safer_route([short_risky, long_safe], max_detour=1.4)
    assert best is long_safe and "0 crossings without signals instead of 1" in why and "longer" in why


def test_fixture_replay_never_hits_network(tmp_path):
    with pytest.raises(FixtureMissing):
        Http(tmp_path, mode="replay").request("GET", "https://example.org/x", params={"a": 1})


def test_fixtures_contain_no_keys():
    for f in default_fixture_dir().glob("*.json"):
        text = f.read_text()
        assert "AIza" not in text and "X-Goog-Api-Key" not in text and "MODEL_API_KEY" not in text


# Navigator -------------------------------------------------------------------

def walk(route, cfg=None, **track):
    nav = Navigator(route, "the gallery", cfg=cfg)
    return nav, run_walk(nav, synth_track(route, TrackConfig(**track)))


def test_dc_walk_cues_in_order_and_never_permit_crossing(dc_routes):
    route, _ = choose_safer_route(dc_routes)
    nav, cues = walk(route, detour_at_m=280.0)
    kinds = [c.kind for c in cues]
    assert kinds[0] == "route_start" and kinds[-1] == "arrive"
    assert kinds.count("crossing_preview") == 4 and kinds.count("at_crossing") == 4
    assert "off_route" in kinds and kinds.index("back_on_route") > kinds.index("off_route")
    for c in cues:
        assert not permits_crossing(c.text), c.text
        assert "\u2014" not in c.text
    at = [c.text for c in cues if c.kind == "at_crossing"]
    assert all("Wait for my check" in t for t in at)


def test_off_route_threshold_grows_with_bad_gps():
    r = straight_route(300)
    cfg = NavConfig()
    nav = Navigator(r, cfg=cfg)
    nav.start()
    p = geo.destination(geo.destination(FROM, 180, 60), 90, 25)
    cues = []
    for t in range(5):
        cues += nav.update(Fix(t=t, lat=p[0], lon=p[1], accuracy_m=30.0))
    assert not any(c.kind == "off_route" for c in cues)
    assert any(c.kind == "gps_weak" for c in cues)
    nav2 = Navigator(r, cfg=cfg)
    nav2.start()
    cues2 = []
    for t in range(5):
        cues2 += nav2.update(Fix(t=t, lat=p[0], lon=p[1], accuracy_m=5.0))
    off = [c for c in cues2 if c.kind == "off_route"]
    assert off and "o'clock" in off[0].text and "GPS" in off[0].text


def test_camera_hint_fuses_map_side_with_camera_without_permission():
    r = straight_route(200, crossings=(60,))
    nav = Navigator(r)
    nav.start()
    for t, d in enumerate(range(0, 56, 5)):
        p = geo.destination(FROM, 180.0, d)
        nav.update(Fix(t=t, lat=p[0], lon=p[1], accuracy_m=5.0))
    for cw, light in [({"found": True, "zone": "ahead"}, "walk"), ({"found": False}, "unknown"), (None, None)]:
        cue = nav.camera_hint(cw, light)
        if cue:
            assert not permits_crossing(cue.text)
    hint = nav.camera_hint({"found": False}, "unknown")
    assert hint is not None and "signal" in hint.text and "o'clock" in hint.text


def test_unspeakable_route_stays_silent():
    r = straight_route(100, crossings=(40,))
    r.speakable = False
    nav, cues = walk(r)
    assert cues and all(not c.speak for c in cues)


def test_srt_has_one_block_per_spoken_cue(dc_routes):
    route, _ = choose_safer_route(dc_routes)
    _, cues = walk(route)
    srt = to_srt(cues)
    assert srt.count(" --> ") == len([c for c in cues if c.text])
    assert srt.startswith("1\n00:00:00,000 --> ")


# Google Maps Experience agent tools (mocked) -------------------------------------

class FakeMcp:
    def __init__(self, result=None, error=None):
        self.seen: list[tuple[dict, dict]] = []
        self.result, self.error = result, error

    def __call__(self, payload, headers):
        self.seen.append((payload, dict(headers)))
        if payload["method"] == "initialize":
            return {"jsonrpc": "2.0", "id": payload["id"], "result": {}}, {"Mcp-Session-Id": "s1"}
        if payload["method"].startswith("notifications/"):
            return None, {}
        if self.error:
            return {"jsonrpc": "2.0", "id": payload["id"], "error": {"message": self.error}}, {}
        return {"jsonrpc": "2.0", "id": payload["id"], "result": self.result}, {}


PLACES = {"places": [{"displayName": {"text": "Renwick Gallery"}, "accessibilityOptions": {"wheelchairAccessibleEntrance": True},
                      "googleMapsLinks": {"placeUrl": "https://maps.google.com/?cid=1"}}]}


def test_google_agent_tools_protocol_and_sources():
    fake = FakeMcp({"structuredContent": PLACES})
    g = GoogleMapsAgentTools(api_key=FAKE_KEY, post=fake)
    res = g.search_places("Renwick Gallery", near=FROM)
    methods = [p["method"] for p, _ in fake.seen]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    assert fake.seen[-1][1]["Mcp-Session-Id"] == "s1"
    assert fake.seen[-1][0]["params"]["name"] == "search_places"
    assert [s.url for s in res.sources] == ["https://maps.google.com/?cid=1"]
    assert extract_sources({"text": "none"}) == []


def test_google_errors_never_leak_the_key():
    g = GoogleMapsAgentTools(api_key=FAKE_KEY, post=FakeMcp(error=f"bad key {FAKE_KEY}"))
    with pytest.raises(RuntimeError) as e:
        g.search_places("x")
    assert FAKE_KEY not in str(e.value) and "AIza" not in str(e.value)


def test_google_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    assert not GoogleMapsAgentTools().enabled
    s = build_session(MapsConfig(enabled=True, provider="fixture", google=True))
    assert s.google is None and all(t["name"] != "google_maps_search" for t in s.tools)


# MapSession and Wonder ---------------------------------------------------------

def session(google=None):
    s = build_session(MapsConfig(enabled=True, provider="fixture"))
    s.google = google
    s.set_position(*FROM, heading=180.0, accuracy_m=10.0)
    return s


def test_session_tools_from_fixtures():
    s = session()
    near = s.dispatch("find_nearby", {"query": "pharmacy"})
    assert near["places"] and all("clock" in p and "distance_m" in p for p in near["places"])
    plan = s.dispatch("plan_route", {"destination": "Renwick Gallery"})
    assert plan["chosen"]["crossings"] >= 1 and "first_cue" in plan
    st = s.dispatch("route_status", {})
    assert st["destination"] == "Renwick Gallery" and "map context only" in st["next_crossing"]["note"]
    assert s.dispatch("plan_route", {"destination": ""})["error"]
    assert s.dispatch("check_crossing_signal", {}) is None


def test_google_search_capped_per_question():
    s = session(GoogleMapsAgentTools(api_key=FAKE_KEY, post=FakeMcp({"structuredContent": PLACES})))
    assert "google_maps_search" in [t["name"] for t in s.tools]
    assert s.dispatch("google_maps_search", {"query": "Renwick Gallery hours"})["sources_shown_in_app"] == 1
    assert "limit" in s.dispatch("google_maps_search", {"query": "again"})["error"]
    s.new_question()
    assert "google_maps" in s.dispatch("google_maps_search", {"query": "again"})


def _wonder_tools(maps):
    hist = [{"t": 1.0, "objects": [], "signal": "unknown", "crosswalk": {"found": False}, "underfoot": "sidewalk"}]
    return GuideTools(history=lambda s: hist, current=lambda: hist[-1], segment=lambda c: {},
                      crossing=lambda: {"spoken": "Do not cross.", "local_light": "unknown",
                                        "fuse": {"decision": "stop"}, "validated": True}, maps=maps)


def test_wonder_uses_map_tools_and_discloses_google_sources():
    from test_wonder_agent import FakeClient, call, text_resp, tool_resp

    s = session(GoogleMapsAgentTools(api_key=FAKE_KEY, post=FakeMcp({"structuredContent": PLACES})))
    client = FakeClient([
        tool_resp(call("plan_route", '{"destination": "Renwick Gallery"}'),
                  call("google_maps_search", '{"query": "Renwick Gallery accessible entrance"}')),
        text_resp("Head south, about 460 metres. The gallery has a step free entrance. You can cross now."),
    ])
    res = WonderGuide(WonderConfig(enabled=True), client=client).run("Take me to the Renwick Gallery", np.zeros((8, 8, 3), np.uint8), _wonder_tools(s))
    names = {t["name"] for t in client.calls[0]["tools"]}
    assert {"plan_route", "route_status", "find_nearby", "google_maps_search"} <= names
    assert "cross now" not in res.speech.lower()
    assert res.speech.endswith(DISCLOSURE)
    assert res.sources and res.sources[0]["url"].startswith("https://maps.google.com")
    assert s.nav is not None


def test_wonder_without_maps_has_no_map_tools():
    from test_wonder_agent import FakeClient, text_resp

    client = FakeClient([text_resp("Sidewalk ahead.")])
    WonderGuide(WonderConfig(enabled=True), client=client).run("what is ahead", np.zeros((8, 8, 3), np.uint8), _wonder_tools(None))
    assert "plan_route" not in {t["name"] for t in client.calls[0]["tools"]}


def test_maps_config_is_opt_in(monkeypatch):
    for k in ("STREETSENSE_MAPS", "STREETSENSE_MAPS_GOOGLE", "STREETSENSE_MAPS_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert not MapsConfig.from_env().enabled
    monkeypatch.setenv("STREETSENSE_MAPS", "1")
    assert MapsConfig.from_env().enabled


def test_walk_sim_cli_offline_without_clip(tmp_path):
    from streetsense.cli import main

    rc = main(["walk-sim", "--clip", "", "--out-dir", str(tmp_path), "--name", "w"])
    assert rc in (0, None)
    lines = (tmp_path / "w.jsonl").read_text().splitlines()
    summary = json.loads(lines[0])["summary"]
    assert summary["chosen"]["crossings"] == 4 and summary["crossing_checks"] == []
    assert (tmp_path / "w.srt").read_text().strip() and "OpenStreetMap" in (tmp_path / "w.md").read_text()
