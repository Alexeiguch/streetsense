#!/usr/bin/env bash
# Download the extra clips for the broader Muse crossing eval (eval/muse_labels_broad.csv)
# into samples/ext/ (gitignored). Sources and licences: eval/SOURCES.md.
# Checks sha256 so the labelled frame times still match.
set -euo pipefail
cd "$(dirname "$0")/../samples"
mkdir -p ext && cd ext
UA="streetsense-eval/0.1 (+https://github.com/Alexeiguch/streetsense)"
WM=https://upload.wikimedia.org/wikipedia/commons
get() { # name url
  [ -s "$1" ] || { curl -fL -A "$UA" -o "$1" "$2"; sleep 1; }
}
get most_11_day.webm   "$WM/0/03/Most%2C_t%C5%99._Budovatel%C5%AF%2C_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._%2811%29.webm"
get most_02_dusk.webm  "$WM/7/76/Most%2C_t%C5%99._Budovatel%C5%AF%2C_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._%2802%29.webm"
get most_03_dusk.webm  "$WM/1/1b/Most%2C_t%C5%99._Budovatel%C5%AF%2C_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._%2803%29.webm"
get most_04_dusk.webm  "$WM/b/b8/Most%2C_t%C5%99._Budovatel%C5%AF%2C_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._%2804%29.webm"
get turkey_beacon_a.webm "$WM/6/67/Red_green_flashing_pedestrian_crossing.webm"
get turkey_beacon_b.webm "$WM/e/e1/Red_red_flashing_pedestrian_crossing.webm"
get tirana_dusk.webm   "$WM/2/2c/Tirana_traffic_lights.webm"
# Pexels (Pexels License). /download/video/<id>/ redirects to the file we used.
for pair in 30404309:pexels_30404309_eu_ped.mp4 5665107:pexels_5665107_eu_ped.mp4 \
            30528306:pexels_30528306_eu_ped_side.mp4 17721925:pexels_17721925_night_ped.mp4 \
            4169275:pexels_4169275_night_ped_jp.mp4 5220221:pexels_5220221_night_hand.mp4 \
            7251025:pexels_7251025_night_square.mp4 5514350:pexels_5514350_us_ped.mp4; do
  get "${pair#*:}" "https://www.pexels.com/download/video/${pair%%:*}/"
done
sha256sum -c --quiet <<'SUMS'
f62a0cb47856b0570620aed64eecc33b15aec9d45e354f6bcef3ce1d0ca9137e  most_02_dusk.webm
6bfc0d097a5c97468462ba48139737aad79c36f067d2f819ee642068c6de0e2a  most_03_dusk.webm
0d77b81c7670ff7766240766a83a7fab4a3ad3aeb81d72f039113285b0acf423  most_04_dusk.webm
06a3af141481155a43f198c7068fd1af87f1141b0e0c989be92d6e094e1953f0  most_11_day.webm
b879c098ab34799f5bff80593a48c306bd32cc71a390e585139e7a33905cfb09  pexels_17721925_night_ped.mp4
6a90acd49449efe0d0d49fa06c4deac7f783a2aa547d9bfea554243adff74458  pexels_30404309_eu_ped.mp4
37d33143600c15dbc86eba46648a1cfc80b8b5232dbca28bd4e8189505f2a439  pexels_30528306_eu_ped_side.mp4
1933794a2929542043ca46ad463662dc3ab97c9f1fedb107138da4452ad44d89  pexels_4169275_night_ped_jp.mp4
a81ac54febbdc0b4b3780e923f51f7570002c596efed010f959051cf2f3bd511  pexels_5220221_night_hand.mp4
fcea1d7a254ee238a57f69330471213e9e0779fc44ba7617dddaa9f7306aacd4  pexels_5514350_us_ped.mp4
5335cccc0902ae4b8a52b7168cbea5d1b351caec139c2a3b17be0d3654ef7ffa  pexels_5665107_eu_ped.mp4
5e8ed341c3be7985d41c7b7957fd0bf99f606d578995171fb549c1fb0088062c  pexels_7251025_night_square.mp4
96b93c02103bbd42ea8b08d3f6c1ea2d6c6607ea426a92e129abcb6ab3b0a6ea  tirana_dusk.webm
2ca8c79248e01b1d89c85a6da5fe10d8b62927baf82cd85316b5386484abec84  turkey_beacon_a.webm
8a568f78865a43c4e4d80765987b894f7eaf7e1c476120bba859c97d33752be3  turkey_beacon_b.webm
SUMS
ls -la
