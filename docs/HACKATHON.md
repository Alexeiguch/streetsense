# Meta Global AI Developer Hackathon: key facts

Sources: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details (event page plus the Official Rules / waiver text embedded in it), Cerebral Valley emails to Alex (26 Oct 2026), Meta Model API docs at https://dev.meta.ai/docs. Checked 26 Oct 2026.

## Dates
- Hackathon: Mon 26 Oct 12:00 PM ET to Wed 4 Nov 9:00 PM ET (remote).
- **Submission deadline: Wed 4 Nov, 9:00 PM ET = Thu 5 Nov, 02:00 UTC/London.** The platform data shows `submissionDueDateGracePeriodHours: 6`, but nothing public explains it, so don't count on it.
- Workshops and Discord office hours run 27 Oct to 4 Nov. Judging runs 4 to 8 Nov. Live finals and winners on 9 Nov. Bonus prizes are announced around 10 Nov.
- Meta Model API credits: $150 per participant, valid for a 10-day window. Claim them through the form by 4 Nov 9 PM ET. A credit card has to be on the account before API calls work, so set a spend-limit alert on the Billing page.

## Tracks (enter exactly one)
- **Indie Developer** (individuals or students, teams of up to 4): $400k total ($150k / $100k / $75k / $50k / $25k). Judged on creativity, technical ambition, and whether it works.
- **Startup** (technical founders): $550k total. Judged on technical execution and whether it holds up in production.
- Bonus prizes up to $50k (community vote, most viral, and others).

## Themes
- **Build for Your World**: bring Meta's models into your own domain and solve a problem you know firsthand. This project fits here (accessibility).
- **Push the Frontier**: software, robotics, physical systems, and similar.

## Judging criteria (from the Official Rules)
| Criterion | Weight | Meaning |
|---|---|---|
| Impact | 30% | Real-world potential, who benefits, whether people would use it, fit with a problem statement |
| Use of Meta AI tooling | 25% | Creative use of Meta's tooling beyond a basic integration; surfacing unexpected capabilities |
| Technical depth & execution | 20% | Sound, refined engineering |
| Demo | 25% | A working, impressive demo that holds up live, plus the quality of the demo video |

## Required / expected Meta tech
The event now centres on the **Muse** model family through **Meta Model API**, not Llama:
- `muse-spark-1.3`: vision (image, video, audio, and PDF input). OpenAI-compatible at `https://api.meta.ai/v1`, key in env var `MODEL_API_KEY`. Images go in **user** messages only (a system or developer image returns HTTP 400). Supports structured output, tool calling, and "perception grounding" (coordinates as text).
- `muse-wonder-1.0`: the most capable agentic/coding model (preview).
- `sam-3.1` (**Segment Anything 3.1**): text prompt ("crosswalk", "curb") plus image or video, returning boxes and pixel masks, with tracking across video frames. Called through the Responses API.
- `muse-voice-transcribe-1.0`: realtime speech-to-text over WebSocket (could handle voice commands like "can I cross?").
- Open weights: **Muse Glimmer** (multimodal, distilled from Spark, runs on your own hardware with vLLM, llama.cpp, or ExecuTorch, so it could run on-device) and Muse Spark Open (816B MoE).
- The Rules say "build with Meta AI products, including Muse Spark, Glimmer, SAM 3, and others", but **no hard minimum is stated**. Use of Meta tooling is still 25% of the score.
- Meta's cookbook includes a **Ray-Ban Meta glasses** recipe ("VisionClaw" with OpenClaw + Muse Spark): https://dev.meta.ai/docs/cookbook/rayban-openclaw.md. It's directly relevant for a glasses version of this project.
- Partner credits (selected participants only): Baseten $100, Modal $100, Fireworks $25, all for open-weight models.

## Submission
- Through the portal at https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit (needs login; we couldn't see the form fields).
- Indie track: one Representative submits and names all teammates.
- A demo video is expected (it's part of the Demo score). **Length, repo requirement, and required written fields aren't visible publicly.** Check the submit form or Discord.

## Pre-existing code / IP
- The submission must be the entrant's original work. Third-party technical help is allowed if you own the result. Open-source software is allowed if its licences are followed and the submission "enhances or builds upon" it.
- **The public rules say nothing explicit about code written before 26 Oct.** Ask on Discord if it matters. This repo was started on 26 Oct 2026, after kickoff.
- Licence note: Ultralytics YOLO is **AGPL-3.0**. That's fine for a hackathon and open source, but a closed commercial product would need an Ultralytics licence or a different detector.
- Entrants grant the organisers a perpetual, non-exclusive licence to use the submission for marketing and similar purposes.

## What we could not see
- The problem statements list. Judging mentions "fits one of the problem statements", but the public page only shows the two themes. It may sit behind login, on Discord, or in the kickoff stream.
- The submit form fields, video length limit, and whether a public repo is needed.
- Discord content and the kickoff recording.
