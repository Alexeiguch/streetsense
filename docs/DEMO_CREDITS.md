# StreetSense demo — credits and evidence

## Product and model outputs

[StreetSense](https://github.com/Alexeiguch/streetsense), local working revision based on `cf1d91cf7a9be910718e7af659a3ff51dc95e3d3`. Original source code is AGPL-3.0; this does not replace the licenses of third-party assets, models or media.

The local footage was processed by the actual current pipeline on 4 November 2026, with remote calls disabled. Captures include detections, tracking IDs, local readings and selected spoken alert text. `captures/*.json` records source hashes and frame indices. These are edited recordings of offline analysis, not a live camera-navigation trial.

Muse Spark 1.3 and SAM 3.1 material comes from actual evaluations on 4 November, recorded in `docs/review_2026_11_04/live-meta.json`. Each Muse result is displayed with its original input frame. SAM previews are actual decoded masks. The renderer does not invent or regenerate their decisions. No new Meta API calls were made to produce this film.

Crossing cue text is the exact output of the production validation/fusion/speech renderer. The vehicle alert is the exact selected local output at Tyumen capture row 59 (source time 3.9333s). Both are revoiced with a stock neural voice for the recording. It is not device audio. The expiry scene is explicitly a rule replay with an 8-second maximum age, not a new model evaluation or live crossing.

## Footage

| Material | Author and source | License |
|---|---|---|
| DC signal examples | Sebastian Wallroth, [Pedestrian crossing light washington dc](https://commons.wikimedia.org/wiki/File:Pedestrian_crossing_light_washington_dc.ogv) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| Tyumen crossing; Muse disagreement; SAM preview | RG72, [Piedirejo en strato Herzen (Tjumeno)](https://commons.wikimedia.org/wiki/File:Piedirejo_en_strato_Herzen_(Tjumeno).webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| European-style pedestrian signal | Davlat Tursunov, [Pexels 6415590](https://www.pexels.com/video/6415590/) | [Pexels License](https://www.pexels.com/license/) |
| Tomaszów sidewalk/scooter | WrS.tm.pl, [A person using a public electric kick scooter in Tomaszów Mazowiecki. Saint Anthony Street. October 2021](https://commons.wikimedia.org/wiki/File:A_person_using_a_public_electric_kick_scooter_in_Tomasz%C3%B3w_Mazowiecki._Saint_Anthony_Street._October_2021.ogv) | Public domain, as recorded in the project source manifest |
| Frankfurt vehicle signal | ManuelB701, [FFM – Münchener Straße, Grünschaltung Auto- und Straßenbahnampel](https://commons.wikimedia.org/wiki/File:FFM_-_M%C3%BCnchener_Stra%C3%9Fe,_Gr%C3%BCnschaltung_Auto-_und_Stra%C3%9Fenbahnampel.webm) | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) |

Changes: excerpts, cropping, resizing, annotations, model masks, edited timing, typography, new narration and music. No endorsement is implied. Tyumen-derived annotated segments retain CC BY-SA 4.0. Preserve their attribution and share-alike notices when redistributing those adaptations. Other source material retains its listed license; an AGPL code license does not relicense it. Source credits also appear in the film end card and MP4 metadata.

## Voice, music and typography

- Local [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M) stock voices `bm_george` (British English narration and spoken cue text), rendered via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). No voice cloning or custom identity. Model weights Apache 2.0; runtime MIT. Weights are not redistributed.
- Original score, percussion and transition accents synthesized in `source/sound.py`; no music samples or stock music.
- [Manrope](https://github.com/google/fonts/tree/main/ofl/manrope), SIL Open Font License 1.1, bundled with its full license.
- Python/Pillow/OpenCV composition; FFmpeg H.264/AAC encoding and two-pass loudness normalization. Core video, audio and graphic production runs locally.

## Design reference

The user-provided reference video informed the charcoal palette, restrained typography, blurred-footage title card and pacing. Its footage and audio are not redistributed or reused. This three-minute voice revision uses the stock British English voice bm_george. No voice cloning is performed; subjective listening review remains pending.

## Review

Actual encoded frames, full decode, frame counts, dimensions, duration, captions, stream codecs, loudness, true peak and MP4 faststart are checked in `verification.json`. Captions use phrase timing estimated from generated speech durations, not word alignment. Browser playback is checked separately. This host provides no listening-review tool, so subjective voice/music review remains for the viewer; objective audio checks are recorded.
