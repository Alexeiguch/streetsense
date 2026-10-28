# Maps: walking navigation for StreetSense

Status, 28 Oct 2026: built and tested offline, with recorded OpenStreetMap data for a real Washington DC route.
Google Maps support is written but has **not been run live**, because no `GOOGLE_MAPS_API_KEY` is set on the dev box.
Nothing in this file ran on Ray-Ban glasses. The walk is simulated: a synthetic GPS track along a real route, with a real street clip at each kerb.

## The one rule that does not change

Map data never authorises a crossing. The go rule stays as before: the local reader says walk AND a validated Muse Spark reply says yes (`crossing_answer` in `streetsense/answers.py`).
At a kerb the navigator only says where the crossing is and what the map knows about it. Then it says "Wait for my check before you step out."
Every navigator cue is tested against the same forbidden phrase list that cleans Wonder's speech (`tests/test_maps.py`).

GPS in a city is off by 5 to 15 metres, and by more between tall buildings. That is wider than a crosswalk. So the map gives context (which street, which side, signals or not), and the camera and Muse confirm what is actually there.

## Ideas, ranked by impact for one week of work

| # | Idea | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | **Crossing preview.** 45 m before each crossing, say the street, the lanes and what the map knows: signals, sound, vibration, push button, tactile paving, island. At the kerb, tell the walker what to feel for and to wait for the check. | High: blind walkers say the unknown crossing is the worst part of a route | Low | Built |
| 2 | **Safer route choice.** Ask the router for alternatives and pick the one with the fewest unsignalised crossings, no stairs and least risk, within 1.4x of the shortest. Say why in one sentence. | High | Low | Built |
| 3 | **Clock-face turn-by-turn that is honest about GPS.** "In 40 metres, turn right." Off-route warnings use a threshold that grows with GPS error and always say "GPS can be off by N metres, so stop and check." | High | Medium | Built |
| 4 | **Camera plus map at the kerb.** The map says the crossing is at 11 o'clock; the camera sees a crosswalk ahead. Say "The crosswalk you need is ahead on your left", or warn when they disagree. If the map shows a signal the camera cannot see, say "Turn slowly toward 11 o'clock." | High: this is the Meta part, the glasses camera grounds the map | Low | Built |
| 5 | **Muse Wonder map tools.** "Take me to the Renwick Gallery", "Is there a pharmacy near me?", "How far to my next crossing?" Wonder calls `plan_route`, `find_nearby`, `route_status`. | Medium-high | Low | Built, ran live once |
| 6 | **Google place facts in Wonder.** Opening hours and step-free entrance from Google Maps, through Google's Maps Experience agent tools, with the required spoken disclosure and links in the app. | Medium | Low | Built, mocked only |
| 7 | **Landmarks.** "Bollards ahead", "Gate", "Take the stairs", from Valhalla's blind-user mode. | Medium | Free with 1 | Built |
| 8 | **Where am I.** Nearest address and last street, in one sentence. | Medium | Low | Built (`Navigator.where_am_i`) |
| 9 | Google Navigation SDK in the native Ray-Ban companion app, for Google's own spoken walking directions. | High | High (native app) | Not built, see below |
| 10 | Street-level photos of the next crossing from Mapillary (CC BY-SA) so Muse can describe it before you arrive. Not Street View, see the terms below. | Medium | Medium | Not built |

## What is built

```
streetsense/maps/
  geo.py           distances, bearings, clock face, polylines, snapping to a route
  model.py         Route, Step, Crossing, Place; crossing risk classes; choose_safer_route
  http.py          small HTTP client with record and replay fixtures (headers, so keys, are never stored)
  osm.py           Valhalla (pedestrian, type blind, 2 alternates), Overpass crossing tags, Nominatim search
  navigator.py     GPS fix in, spoken cues out; camera_hint; status; where_am_i
  walk_sim.py      synthetic GPS track with noise, kerb pauses and a detour; SRT subtitles
  google_agent.py  Google Maps Experience agent tools client (MCP over HTTP), sources, disclosure
  session.py       MapSession: the tools Muse Wonder can call
  demo.py          the walk-sim command
```

Crossing data comes from OpenStreetMap tags such as `crossing=traffic_signals`, `traffic_signals:sound=yes`, `traffic_signals:vibration=yes`, `button_operated=yes`, `tactile_paving=yes`, `crossing:island=yes`, the road name and `lanes` from the road it crosses. Valhalla's blind mode adds "Pass traffic signals on I Street" and landmark steps.

### The recorded demo walk

Farragut Square north to the Renwick Gallery, Washington DC, the same area as `samples/dc_ped_light.ogv`.
Valhalla's first route is 919 m with 6 crossings (2 without signals) and 2 sets of stairs. StreetSense picks route 2 instead: "0 crossings without signals instead of 2, 4 crossings instead of 6, no stairs, 382 m shorter".

Recorded output with Muse checks: `docs/maps/walk_sim_muse.md` (transcript), `.srt` (subtitles for the video), `.png` (route plot), `.jsonl` (every cue and check). Highlights:

```
  0s route_start      Route to the Renwick Gallery: 540 metres, about 7 minutes, 4 crossings, all with signals. Walk south.
  5s crossing_preview Crossing in 45 metres at 12 o'clock: K Street Northwest, signals, tactile paving, 1 lane. Stop at the kerb and ask me to check.
 29s at_crossing      At the crossing for K Street Northwest. The map shows signals. Wait for my check before you step out.
 30s camera           The crosswalk you need is ahead.
 32s crossing_check   Do not cross.
161s crossing_preview Crossing in 45 metres at 12 o'clock: I Street Northwest, signals, sound, tactile paving, 4 lanes. ...
269s off_route        You may be off the route. It is about 20 metres away at 4 o'clock. GPS can be off by 10 metres, so stop and check.
296s back_on_route    Back on the route.
347s camera           I see a crosswalk ahead, but the map puts your crossing at 2 o'clock. Check with your cane.
468s landmark         Bollards ahead.
505s arrive           Arriving at the Renwick Gallery, about 10 metres at 12 o'clock.
```

Honest notes on that run:
- The same DC clip is replayed at every kerb (red phase at 55 s, walk phase at 44 s), so the camera is not really at K Street.
- Muse checks ran at the first two kerbs (`--muse-max 2`). At the second kerb the local reader saw walk but Muse said red, so the fusion said "Do not cross." That is the rule working: any disagreement means stop.
- Kerbs 3 and 4 had no Muse check, so the answer is "I can't tell, please wait". Without Muse the go rule cannot pass.

## How to run

All offline by default, from recorded OpenStreetMap responses in `tests/fixtures/maps/`:

```bash
# simulated walk, spoken cues printed, files in outputs/
python -m streetsense walk-sim --plot
# same, with the Muse Spark check at the first 2 kerbs (about $0.003 each)
STREETSENSE_META_BUDGET_USD=2.19 python -m streetsense walk-sim --plot --ask-muse --name walk_sim_muse
# live OpenStreetMap servers, any route (Overpass can take 10 to 40 s)
python -m streetsense walk-sim --provider osm --origin 51.5079,-0.1281 --dest 51.5101,-0.1340 --dest-name "Piccadilly Circus" --clip ""
# Muse Wonder with map tools
python -m streetsense guide --source samples/dc_ped_light.ogv --at 44 --maps \
  --question "Take me to the Renwick Gallery. How far is it and what crossings are on the way?"
```

Options: `--detour-at -1` for no detour, `--noise` and `--accuracy` for GPS quality, `--clip-at 55,44` for the clip second at each kerb.
In code: `STREETSENSE_MAPS=1` turns on `MapsConfig`; `streetsense.maps.build_session()` returns a `MapSession` to pass as `GuideTools.from_pipeline(..., maps=session)`.
Public servers: `STREETSENSE_VALHALLA_URL`, `STREETSENSE_OVERPASS_URL` (comma separated mirrors), `STREETSENSE_NOMINATIM_URL`. For real use, run your own Valhalla and Overpass; the public ones are for light testing.

Live Wonder run (28 Oct 2026): Wonder called `plan_route` once and said "Head south to Renwick Gallery, about 465 metres and 6 minutes. Route has 4 crossings, all signalised." 23 s, $0.013.

## Google Maps: what the terms allow (checked 28 Oct 2026)

Google Maps Platform Terms, section 3.2.3, and the service-specific terms:
- (a)(iv): no "use with text-to-speech services". So raw Routes, Places or Geocoding results cannot be read aloud. That rules out a spoken turn-by-turn built on the Routes API.
- (c)(vi): no turning text driving times into synthesized speech.
- (d)(iv): no re-creating Google navigation. (e): no Google content on a non-Google map.
- (c)(vii): no using Google content to "train, test, validate or fine-tune" AI models. (c) also forbids creating content from Street View images. So Street View frames must not be fed to SAM 3.1 or Muse to find crosswalks.
- (a)(i): no pre-fetching or storing beyond the caching rules (coordinates up to 30 days, place IDs indefinitely).

What is allowed, and what we use:
- **Maps Experience agent tools** (formerly Grounding Lite): Google's MCP server for LLM agents, `https://mapstools.googleapis.com/mcp`, tools `search_places` and `compute_routes`. It is built for third-party models. Conditions: the model must not train on the inputs (Muse Wonder on the Model API standard mode does not), the Google Maps sources must be shown right after the answer, and for voice apps there must be a companion screen with the transcript and sources, a spoken disclosure at the time, and a persistent one.
  StreetSense adds "Some of this comes from Google Maps. Sources are in the app." to the spoken answer, and returns the links in `GuideResult.sources` for the companion screen.
  It is capped at 1 Google call per question.
- **Navigation SDK** (Android and iOS) supports walking with Google's own voice guidance. It is the compliant way to get Google spoken turn-by-turn in the native Ray-Ban companion app. Not built.

So the spoken route, crossing previews and safer route choice use OpenStreetMap (ODbL), which may be spoken and processed with attribution. Google answers place questions inside Wonder.

## Enabling Google (for Alex)

1. In Google Cloud console, pick or create a project with billing on.
2. Enable **Maps Experience agent tools** (the Maps Grounding Lite API). Nothing else is needed for StreetSense today.
3. Create an API key. Under API restrictions, allow only that API. Under application restrictions, use IP addresses for a server, or none for a short demo. Or use Google's free Demo Key for testing.
4. `export GOOGLE_MAPS_API_KEY=...` (never commit it), then add `--google-maps` to `guide --maps`, or set `STREETSENSE_MAPS_GOOGLE=1`.
5. Optional later: the Navigation SDK in the native app, with its own Android or iOS key restrictions.

The key is sent only in the `X-Goog-Api-Key` header, is masked in error messages, and is never written to fixtures or logs.

## Cost per walk

| Part | Cost |
|---|---|
| OpenStreetMap routing, crossings, places (public servers or self-hosted) | $0 |
| Google Maps Experience agent tools, per call | about $0.007 (Essentials), first 10,000 a month free |
| Muse Spark crossing check, per kerb you ask at | about $0.0015 to $0.003 |
| Muse Wonder question with map tools | about $0.008 to $0.017 |

A 4-crossing walk with a check at every kerb and one Wonder question: about $0.02 to $0.03.

Meta spend for this work: about $0.016 (2 Spark checks plus 1 Wonder question).

## Limits

- GPS is simulated. Real phone GPS will jump more near tall buildings; the off-route rule waits for 3 bad fixes and scales with reported accuracy.
- Heading comes from movement, not a compass, so it is unknown while standing still at first.
- OpenStreetMap crossing tags are only as good as local mappers. Unknown crossings are scored as risky and described as "no details on the map".
- Public Valhalla, Overpass and Nominatim servers are rate limited and sometimes overloaded. Overpass has 3 mirrors and one query per route set.
- Not wired into the web demo or the glasses companion app yet.
