Eval runs for codex/signal-reacquisition review (Oct 29-30 2026).
All: signal CNN v1 (default), detector models/yolo11s_openvino_model (OpenVINO CPU), STREETSENSE_SAM_FALLBACK=0, no Meta calls.
Command: python scripts/eval_local_reader.py --dense --out <file>.json
Tiles runs add STREETSENSE_LIGHT_TILES=2x3 STREETSENSE_LIGHT_TILES_MODE=always.
sr_main = 5b70b28 (main). sr_merged = 7f41d7c = 5b70b28 + eb13a27 (local merge, not pushed).
eval2_sr_main_notiles: main, no tiles. eval3_sr_merged_notiles: merged, no tiles (held-out loss: heldout_eu 4/7 -> 0/7, dev night 2/3 -> 1/3).
eval3_sr_main_tiles / eval3_sr_merged_tiles: tiles 2x3, totals equal.
