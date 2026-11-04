# Meta Global AI Developer Hackathon: key facts

Sources: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details and the official waiver, organizer emails through 4 Nov 2026, the signed-in submission form, and Meta Model API docs at https://dev.meta.ai/docs. Submission requirements rechecked 4 Nov 2026. See [the submission checklist](SUBMISSION_2026_11_04.md).

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

## How applications are reviewed (application form only)

Separate from the submission judging below. Quoted from the organisers:

We aim for a mix of technical backgrounds and domains across participants. Applicants should best showcase their technical skills in the application form. We review every application against two criteria, using only your answers on the application form:

1. **Your project idea:** how clear and feasible your proposed project is, and how well it fits one of the two project themes.
2. **Your technical skill:** the technical skill shown in the AI project you describe, including what you built and the techniques you used. Links to your GitHub, LinkedIn or a shipped product help us verify your experience and skill but are not required.

Both criteria are weighted equally. Applications are evaluated on a rolling basis. Eligible applications meeting minimum scoring cut-offs will be accepted automatically, up to the first 20,000 acceptances. Eligible applications that do not meet the minimum scoring cut-offs will be accepted based on score received, if space allows. Answers to questions 13 to 16 help us improve our developer programs and are not used in selection.

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
- **3-Minute Demo Video** link and **Public GitHub Repository** are explicit fields in the signed-in form. The 3 Nov organizer email also says the GitHub must be open source, mandatory for all projects.
- Team name/members, track, problem statement, project description (3,000 characters), use of Meta products (3,000), why Meta's models matter (3,000), and partner feedback (150) appear in the form, followed by survey questions and a permission checkbox. See the checklist for the exact prompts.
- Optional Most Viral entry: publish a demo on X and provide its post link in the form. The 4 Nov organizer email announces a $20,000 prize for engagement through 8 Nov 11:59 PM ET (9 Nov 04:59 London).

## Pre-existing code / IP
- The submission must be the entrant's original work. Third-party technical help is allowed if you own the result. Open-source software is allowed if its licences are followed and the submission "enhances or builds upon" it.
- **The public rules say nothing explicit about code written before 26 Oct.** Ask on Discord if it matters. This repo was started on 26 Oct 2026, after kickoff.
- Licence note: Ultralytics YOLO is **AGPL-3.0**. That's fine for a hackathon and open source, but a closed commercial product would need an Ultralytics licence or a different detector.
- Entrants grant the organisers a perpetual, non-exclusive licence to use the submission for marketing and similar purposes.

## What we could not see
- The problem statements list. Judging mentions "fits one of the problem statements", but the public page only shows the two themes. It may sit behind login, on Discord, or in the kickoff stream.
- The exact external video-host restrictions, if any. The form presents a link field but no visible hosting or file-size specification. Use a judge-accessible link and verify playback.
- Discord content and the kickoff recording.
