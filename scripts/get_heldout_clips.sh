#!/usr/bin/env bash
# Fetch the held-out reader clips into samples/heldout/ (gitignored). See eval/SOURCES.md.
# Includes the three exact still photos, with source links in eval/SOURCES.md.
set -euo pipefail
cd "$(dirname "$0")/../samples" && mkdir -p heldout && cd heldout
UA="streetsense-eval/1.0 (research; contact via repo)"
get() { [ -s "$1" ] || curl -fsSL -A "$UA" -o "$1" "$2"; }
WM=https://upload.wikimedia.org/wikipedia/commons
get uk_green.jpg "$WM/8/85/Green_man_uk.jpg"
get uk_red_shrews.jpg "$WM/2/20/Red_man_at_pelican_crossing_on_Mytton_Oak_Road%2C_Shrewsbury%2C_August_2025.jpg"
get basel.jpg "$WM/4/46/UEFA_Women%27s_Euro_Basel_traffic_light.jpg"
get nz.ogv           "$WM/0/05/Pedestrian_signals_in_New_Zealand.ogv"
get ffm_vehicle.webm "$WM/7/7b/FFM_-_M%C3%BCnchener_Stra%C3%9Fe%2C_Gr%C3%BCnschaltung_Auto-_und_Stra%C3%9Fenbahnampel.webm"
get pexels_6415590.mp4 "https://www.pexels.com/download/video/6415590/"
# The FFM clip is AV1, which OpenCV cannot decode here. Re-encode once to h264.
[ -s ffm_vehicle.mp4 ] || ffmpeg -loglevel error -y -i ffm_vehicle.webm -c:v libx264 -pix_fmt yuv420p -an ffm_vehicle.mp4
if command -v shasum >/dev/null; then
  check_sha=(shasum -a 256 -c)
else
  check_sha=(sha256sum -c)
fi
"${check_sha[@]}" <<'SUMS'
73790974cee83cdcf141bc7acb1911a1c44f4d6a5f5328ef3b176c778eeed0f7  uk_green.jpg
aae1c76f77c9b09f653e100e5f38b23344069ad0847d3d472457d41f12b3a3c2  uk_red_shrews.jpg
ce4818b7d7292728d9c917122e53ff63ebb0a2eb735f53301f6fe7343005dade  basel.jpg
7f2bc8e1305232ba04c6cee052aa6ea29ecf40ebec8da7eccaf55a7cae8e7d5c  nz.ogv
28a4c3402174fe6f536e8bc9bef0ac2001e088e74d34a474181a04e9ebc5d010  ffm_vehicle.webm
89bd7db2fc30645770803d73cb65c2a5e1f3be9eef73a78c7000ce7935e9be3a  pexels_6415590.mp4
SUMS
ls -la
