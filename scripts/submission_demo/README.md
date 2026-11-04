# Demo production source

These scripts produced the three-minute 1920×1080 judge film and 50-second 1080×1920 social edit on 4 November 2026. Both use real recorded StreetSense outputs, original-frame Muse/SAM evaluations, large captions, stock local neural narration and original synthesized music. The latest revision uses the supplied reference’s charcoal palette, white sentence-case headings and blurred-footage title card, with the product title, author and description at 5 seconds. Narration uses `am_fenrir`, selected by approximate pitch comparison, with `am_michael` product cues. They make no field-safety claim.

The editable media bundle lives in `outputs/submission-demo-v3/` locally. Large captured footage, audio takes and final MP4s are deliberately outside Git. [Media sources, changes and licenses](../../docs/DEMO_CREDITS.md) are documented separately from the AGPL code license.

With that bundle present:

```bash
export FFMPEG=/absolute/path/to/ffmpeg
python scripts/submission_demo/render.py
python scripts/submission_demo/render.py --portrait
python scripts/submission_demo/sound.py
python scripts/submission_demo/export.py
python scripts/submission_demo/verify.py
```

Dependencies: numpy, Pillow, OpenCV and FFmpeg. To revoice, install `kokoro-onnx` and point `KOKORO_MODELS` at the separately downloaded Kokoro v1.0 weights and voices. Bundled WAVs avoid a speech-model download when only editing the picture.

`capture.py` reruns the current local pipeline on the documented public samples, with Meta disabled. `prepare.py` bundles the source images/SAM previews from `scripts/evaluate_review_meta.py`. Those prerequisites must exist first; fresh evaluations use their own documented model/API setup and costs. `timeline.py` is the editable script and shot list; `render.py` contains the landscape and portrait compositions. `STREETSENSE_ROOT` and `STREETSENSE_DEMO_ROOT` can point at another checkout or media-bundle directory. The current master is `StreetSense-hackathon-3min-v3.mp4`; its 149–159s section demonstrates an actual selected vehicle alert.

The `479 passed, 1 skipped` figure is the recorded project test result at production time. Export verification checks the encoded frames, duration, captions, actual AAC loudness and true peak, and whether cue text exactly matches the recorded production renderer. Subjective listening review requires a human playback pass.

The final British-voice judge film and complete editable production bundle are available in the [hackathon release](https://github.com/Alexeiguch/streetsense/releases/tag/hackathon-2026-11-04). Use the bundle’s source/render.py, source/sound.py and source/finish.py to reproduce that final export from the bundled recorded evidence and voice WAVs.
