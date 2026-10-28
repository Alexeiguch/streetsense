#!/usr/bin/env bash
# Fetch the held-out reader clips into samples/heldout/ (gitignored). See eval/SOURCES.md.
# The three still photos (uk_green.jpg, uk_red_shrews.jpg, basel.jpg) are placed by hand;
# their sources and sums are listed in eval/SOURCES.md.
set -euo pipefail
cd "$(dirname "$0")/../samples" && mkdir -p heldout && cd heldout
UA="streetsense-eval/1.0 (research; contact via repo)"
get() { [ -s "$1" ] || curl -fsSL -A "$UA" -o "$1" "$2"; }
WM=https://upload.wikimedia.org/wikipedia/commons
get nz.ogv           "$WM/0/05/Pedestrian_signals_in_New_Zealand.ogv"
get ffm_vehicle.webm "$WM/7/7b/FFM_-_M%C3%BCnchener_Stra%C3%9Fe%2C_Gr%C3%BCnschaltung_Auto-_und_Stra%C3%9Fenbahnampel.webm"
get pexels_6415590.mp4 "https://www.pexels.com/download/video/6415590/"
# The FFM clip is AV1, which OpenCV cannot decode here. Re-encode once to h264.
[ -s ffm_vehicle.mp4 ] || ffmpeg -loglevel error -y -i ffm_vehicle.webm -c:v libx264 -pix_fmt yuv420p -an ffm_vehicle.mp4
sha256sum -c --quiet <<'SUMS'
7f2bc8e1305232ba04c6cee052aa6ea29ecf40ebec8da7eccaf55a7cae8e7d5c  nz.ogv
28a4c3402174fe6f536e8bc9bef0ac2001e088e74d34a474181a04e9ebc5d010  ffm_vehicle.webm
89bd7db2fc30645770803d73cb65c2a5e1f3be9eef73a78c7000ce7935e9be3a  pexels_6415590.mp4
SUMS
ls -la
