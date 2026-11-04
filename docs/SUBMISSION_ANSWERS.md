# Draft submission answers

Prepared from the actual implementation and measurements on 4 November 2026. Review before copying into the [submission form](https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit). Personal survey ratings and the promotion permission are for the submitter to answer.

Team name: **StreetSense**. Suggested track: **Indie Developer Track**. Problem statement: **Build for Your World**. Repository: https://github.com/Alexeiguch/streetsense . Video link: add the hosted three-minute export after checking access.

## Project Description

StreetSense is a camera-based street-scene assistant prototype for blind and low-vision pedestrians. It turns a busy camera feed into short, prioritized spoken cues about crossings, pedestrian signals, moving vehicles and nearby obstacles.

The system has two loops. Local computer vision detects and tracks people, vehicles and signal heads, understands road/sidewalk surfaces, and reads pedestrian signals. Background Meta AI adds scene context and an independent crossing assessment. A deterministic decision layer controls what can be spoken: crossing permission requires both a local WALK reading and a validated positive Muse Spark response. Unclear, conflicting, malformed or stale evidence cannot authorize crossing.

The demo shows recorded operation on real street footage, including a case where Muse Spark says yes but the local reader is unclear and the system refuses a go cue. It also shows a nearly expired countdown, European-style walking figures, and actual Segment Anything masks. Recorded model replies and replayed rule tests are labelled.

Engineering work includes country/style profiles for UK and European signals, bounded signal-head refinement, per-track temporal voting, timestamp-based expiry, generation checks for asynchronous replies, scene-reset handling, prioritized speech and segmentation caching. The current test suite has 479 passing tests and one skipped test.

A separate Meta MapAnything module investigates 3D reconstruction. It produces real point clouds and camera estimates, but current consistency and runtime results do not justify integration into live guidance. The experiment remains offline.

This is a prototype evaluated on public recordings, not a validated mobility aid or a completed field study. Next steps are accessibility-led evaluation and calibrated real-world measurements. Source, reproduction instructions, measured limitations and media credits are available in the repository.

## Describe your use of Meta products to build the submitted project

Muse Spark 1.3 is integrated through Meta Model API for multimodal street-scene reasoning and crossing assessment. The application requests structured fields describing pedestrian signals, countdowns, hazards and whether the model considers crossing appropriate. Responses are validated before use. Free-form model text is not permitted to bypass the crossing renderer.

Meta output is combined with an independently computed local signal reading. The local reader must identify WALK and Muse must return a validated positive result before the crossing renderer can produce its positive cue. Red signals, low countdowns, uncertainty and disagreements block that cue. Capture timestamps, expiry and scene-generation checks prevent late background responses from reusing evidence after a camera rewind or geometry change.

Segment Anything 3.1 supplies text-prompted boxes and pixel masks for objects such as pedestrian signal heads and crosswalks. Its fallback/advisory path is separate from the default fast perception loop. Masks use the correct request-to-display coordinate transform and do not authorize crossing. The demo includes actual recorded SAM responses.

We also implemented a separate offline experiment using Meta's Apache-licensed MapAnything checkpoint. It reconstructs camera poses, depth and colored point clouds from several views. The module checks image correspondences, parallax, invalid geometry, latency and memory, and compares depth shape with the existing Depth Anything V2 baseline under shared cameras. The moving scene shows a modest consistency gain, but failed gates and seconds-per-batch latency keep reconstruction outside the production decision path.

The project uses Meta models as constrained components of a measured system, rather than treating a single model answer as movement authority. The source includes model/API integration, tests, replayable evidence and honest performance reports. No use of Muse Code is claimed.

## Why are Meta's models important to your solution?

Street understanding needs more than object labels. Muse Spark can interpret a pedestrian head, a nearly expired countdown and the surrounding street context in one image. That provides a useful second perspective alongside local computer vision, which is fast but can miss a small, unusual or partly visible signal.

The disagreement case is central to the design. On one recorded street image, Muse Spark returned yes while the local reader remained unclear. The deterministic fusion layer refused a go cue. This shows both the value of richer scene reasoning and the need to constrain it with independent evidence, schema checks and freshness limits.

Segment Anything provides spatial grounding: text such as crosswalk or traffic light can produce actual pixel masks and boxes, which can be inspected and aligned with local geometry. This is useful as an advisory capability without making every camera frame wait for a remote call.

MapAnything lets us test whether multi-view geometry adds useful information, using real output rather than an assumed benefit. The pilot produced an inspectable 3D scene, but its latency and consistency do not yet support the live loop. Keeping that result offline is an engineering decision based on measurements.

Meta Model API's structured, OpenAI-compatible interface also made it practical to share validation, usage accounting and asynchronous request handling. The next step is to establish benefits on calibrated scenes and accessibility-led evaluation rather than extrapolating from a few successful recordings.

## Provide feedback from your experience using partner technologies.

No partner inference services were used; inference ran locally or through Meta Model API.

## Personal survey answers

The remaining form asks about likelihood of building with Meta Model API, overall opinion, recommendation scores for Meta Model API and Muse Code, the biggest obstacle, comparison with other models, selection priorities, switching barriers and production barriers. Answer these from your experience. Do not imply Muse Code was used if it was not.
