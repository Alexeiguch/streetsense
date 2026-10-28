# Eval clip sources

The video files are not committed (`samples/` is gitignored). Fetch them with `scripts/get_samples.sh` (original set) and `scripts/get_eval_clips.sh` (broad set, sha256-checked). The eval extracts frames with ffmpeg at the times listed in `eval/muse_labels*.csv`. The only image committed is the contact sheet `docs/benchmark/muse_eval_broad_labels.jpg`, which shows small crops of the labelled frames.

All clips below were downloaded on 28 Oct 2026. I checked the licences on the source pages that day.

## Original set (`eval/muse_labels.csv`)

See `samples/README.md`. All six clips come from Wikimedia Commons: DC (CC BY 3.0), Tyumen (CC BY-SA 4.0), CDMX (CC BY-SA 4.0), Mazatlán (CC0) and Tomaszów (public domain).

## Broad set (`eval/muse_labels_broad.csv`)

| Clip id | Local file | Source | Author | Licence | What it shows |
|---|---|---|---|---|---|
| `most11` | most_11_day.webm | https://commons.wikimedia.org/wiki/File:Most,_t%C5%99._Budovatel%C5%AF,_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._(11).webm | ŠJů | CC BY 4.0 | Most, Czechia, day. Pedestrian head with green/red man and countdown. |
| `most02` | most_02_dusk.webm | https://commons.wikimedia.org/wiki/File:Most,_t%C5%99._Budovatel%C5%AF,_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._(02).webm | ŠJů | CC BY 4.0 | Same crossing, 21:15, dusk. Two pedestrian heads. |
| `most03` | most_03_dusk.webm | https://commons.wikimedia.org/wiki/File:Most,_t%C5%99._Budovatel%C5%AF,_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._(03).webm | ŠJů | CC BY 4.0 | Same crossing, dusk, 0.9 s. |
| `most04` | most_04_dusk.webm | https://commons.wikimedia.org/wiki/File:Most,_t%C5%99._Budovatel%C5%AF,_odpo%C4%8D%C3%ADt%C3%A1vac%C3%AD_semafory_u_zast%C3%A1vky_1._n%C3%A1m._(04).webm | ŠJů | CC BY 4.0 | Same crossing, dusk. |
| `beacon_a` | turkey_beacon_a.webm | https://commons.wikimedia.org/wiki/File:Red_green_flashing_pedestrian_crossing.webm | Chidgk1 | CC0 | Turkey. Zebra crossing with flashing warning beacons and no pedestrian head. |
| `beacon_b` | turkey_beacon_b.webm | https://commons.wikimedia.org/wiki/File:Red_red_flashing_pedestrian_crossing.webm | Chidgk1 | CC0 | Turkey. Flashing red beacons over a zebra crossing. |
| `tirana` | tirana_dusk.webm | https://commons.wikimedia.org/wiki/File:Tirana_traffic_lights.webm | Lan G. | CC BY 4.0 | Tirana, dusk. Vehicle signals and LED posts. |
| `eu_ped_a` | pexels_30404309_eu_ped.mp4 | https://www.pexels.com/video/pedestrian-traffic-light-changing-sequence-30404309/ | Emre Yıldırım | Pexels License | European-style two-lamp head, day. Location not stated. |
| `eu_ped_b` | pexels_5665107_eu_ped.mp4 | https://www.pexels.com/video/a-pedestrian-traffic-light-turning-green-from-red-5665107/ | Taryn Elliott | Pexels License | Two-lamp red/green man head, day. Location not stated. |
| `eu_ped_side` | pexels_30528306_eu_ped_side.mp4 | https://www.pexels.com/video/traffic-light-changing-at-pedestrian-crosswalk-30528306/ | Evgenij Mikhailov | Pexels License | Side-by-side red/green man lamps, day. Location not stated. |
| `night_ped` | pexels_17721925_night_ped.mp4 | https://www.pexels.com/video/night-shot-cinematic-17721925/ | Mr Clinci | Pexels License | Night. Two-lamp red/green man head. |
| `night_jp` | pexels_4169275_night_ped_jp.mp4 | https://www.pexels.com/video/light-traffic-pedestrian-pedestrian-crossing-4169275/ | Wendy Laplume | Pexels License | Night, 1.2 s. Japanese-style head (red man above, walking man below). |
| `night_hand` | pexels_5220221_night_hand.mp4 | https://www.pexels.com/video/stop-sign-5220221/ | Samson Katt | Pexels License | Night. US red hand, close-up. |
| `night_square` | pexels_7251025_night_square.mp4 | https://www.pexels.com/video/man-walking-at-the-park-during-the-night-7251025/ | MART PRODUCTION | Pexels License | Night. Man at a square, no readable pedestrian signal. |
| `us_ped` | pexels_5514350_us_ped.mp4 | https://www.pexels.com/video/a-pedestrian-traffic-light-5514350/ | "K" (Pexels user) | Pexels License | US walk / countdown / hand head, day. |

Licence links:
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- CC0: https://creativecommons.org/publicdomain/zero/1.0/
- Pexels License: https://www.pexels.com/license/ (free to use and modify, no attribution required, no redistributing the unaltered files on other stock platforms). I credit the authors above anyway.

Caveats:
- "EU" in the results means *European-style heads*. Only the Most clips are known to be in the EU (Czechia). The three `eu_ped_*` Pexels clips do not state a location; the Turkey beacon clips come from Turkey.
- Pexels download URLs redirect to a rendition chosen by Pexels. If the sha256 check fails, Pexels served a different rendition, and the labelled times may need rechecking.
