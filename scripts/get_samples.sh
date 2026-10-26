#!/usr/bin/env bash
# Download the sample street clips used for the demo (Wikimedia Commons, free licences).
set -euo pipefail
cd "$(dirname "$0")/../samples"
UA="streetsense-demo/0.1"
curl -fL -A "$UA" -o tyumen_crossing.webm "https://upload.wikimedia.org/wikipedia/commons/f/f0/Piedirejo_en_strato_Herzen_%28Tjumeno%29.webm"
curl -fL -A "$UA" -o dc_ped_light.ogv "https://upload.wikimedia.org/wikipedia/commons/e/e4/Pedestrian_crossing_light_washington_dc.ogv"
curl -fL -A "$UA" -o cdmx_crossing_720p.webm "https://upload.wikimedia.org/wikipedia/commons/transcoded/5/5b/Cruce_peatonal_de_Eje_Central_L%C3%A1zaro_C%C3%A1rdenas_y_Avenida_Madero%2C_Ciudad_de_M%C3%A9xico.webm/Cruce_peatonal_de_Eje_Central_L%C3%A1zaro_C%C3%A1rdenas_y_Avenida_Madero%2C_Ciudad_de_M%C3%A9xico.webm.720p.vp9.webm"
ls -la
