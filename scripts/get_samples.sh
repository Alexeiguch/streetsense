#!/usr/bin/env bash
# Download the sample street clips used for the demo (Wikimedia Commons, free licences).
set -euo pipefail
cd "$(dirname "$0")/../samples"
UA="streetsense-demo/0.1"

# Crossings / lights
curl -fL -A "$UA" -o tyumen_crossing.webm \
  "https://upload.wikimedia.org/wikipedia/commons/f/f0/Piedirejo_en_strato_Herzen_%28Tjumeno%29.webm"
curl -fL -A "$UA" -o dc_ped_light.ogv \
  "https://upload.wikimedia.org/wikipedia/commons/e/e4/Pedestrian_crossing_light_washington_dc.ogv"
curl -fL -A "$UA" -o cdmx_crossing_720p.webm \
  "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/5b/Cruce_peatonal_de_Eje_Central_L%C3%A1zaro_C%C3%A1rdenas_y_Avenida_Madero%2C_Ciudad_de_M%C3%A9xico.webm/Cruce_peatonal_de_Eje_Central_L%C3%A1zaro_C%C3%A1rdenas_y_Avenida_Madero%2C_Ciudad_de_M%C3%A9xico.webm.720p.vp9.webm"

# Hazards for --open-vocab (scooters / sidewalk clutter)
# CC BY-SA 4.0 — Krzysztof Poplawski
curl -fL -A "$UA" -o katowice_scooter.webm \
  "https://upload.wikimedia.org/wikipedia/commons/a/a6/Hulajnoga_elektryczna_i_licznik_rowerowy_w_dzia%C5%82aniu_w_Katowicach.webm"
# Public domain — WrS.tm.pl
curl -fL -A "$UA" -o tomaszow_scooter.ogv \
  "https://upload.wikimedia.org/wikipedia/commons/9/90/A_person_using_a_public_electric_kick_scooter_in_Tomasz%C3%B3w_Mazowiecki._Saint_Anthony_Street._October_2021.ogv"
# CC0 — DogeGamer2015MZT (sidewalk / kerb / street furniture)
curl -fL -A "$UA" -o mazatlan_sidewalk.webm \
  "https://upload.wikimedia.org/wikipedia/commons/d/db/Video_de_la_Avenida_Ej%C3%A9rcito_Mexicano%2C_Mazatl%C3%A1n%2C_25_de_febrero_de_2022.webm"

ls -la
