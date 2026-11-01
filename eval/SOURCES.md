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

## Held-out reader frames (eval/reader_labels_heldout.csv)

These files are only for checking the local light reader (`scripts/eval_local_reader.py`). None of the reader changes were tuned on them. They live in `samples/heldout/` (gitignored). `scripts/get_heldout_clips.sh` fetches the three clips and re-encodes the FFM clip. The three photos are placed by hand.

| id | file | source | author | licence | notes |
|---|---|---|---|---|---|
| `pex6415590` | pexels_6415590.mp4 | https://www.pexels.com/video/6415590/ ("Close Up Shot of a Pedestrian Light Signal") | Davlat Tursunov | Pexels License | Close-up dotted LED two-lamp head with countdown digits. Red, dark, then green man. |
| `nz` | nz.ogv | https://commons.wikimedia.org/wiki/File:Pedestrian_signals_in_New_Zealand.ogv | Wikimedia Commons uploader (see file page) | CC BY-SA 3.0 | Green man, then flashing red man with countdown. |
| `uk_green` | uk_green.jpg | [Wikimedia Commons, Green man uk.jpg](https://commons.wikimedia.org/wiki/File:Green_man_uk.jpg) | C.Ford | CC BY-SA 3.0 | Still. |
| `uk_red` | uk_red_shrews.jpg | [Wikimedia Commons, Shrewsbury red man](https://commons.wikimedia.org/wiki/File:Red_man_at_pelican_crossing_on_Mytton_Oak_Road,_Shrewsbury,_August_2025.jpg) | Tk420 | CC BY-SA 4.0 | Still. |
| `basel` | basel.jpg | [Wikimedia Commons, UEFA Women’s Euro Basel traffic light](https://commons.wikimedia.org/wiki/File:UEFA_Women%27s_Euro_Basel_traffic_light.jpg) | see file page | CC BY-SA 4.0 | Still. Non-standard figure. |
| `ffm` | ffm_vehicle.webm → ffm_vehicle.mp4 | https://commons.wikimedia.org/wiki/File:FFM_-_M%C3%BCnchener_Stra%C3%9Fe,_Gr%C3%BCnschaltung_Auto-_und_Stra%C3%9Fenbahnampel.webm | ManuelB701 | CC0 | Vehicle and tram heads only. Round green vehicle lamp at 5.3 s; it must never read walk. |

sha256:
```
89bd7db2fc30645770803d73cb65c2a5e1f3be9eef73a78c7000ce7935e9be3a  pexels_6415590.mp4
7f2bc8e1305232ba04c6cee052aa6ea29ecf40ebec8da7eccaf55a7cae8e7d5c  nz.ogv
73790974cee83cdcf141bc7acb1911a1c44f4d6a5f5328ef3b176c778eeed0f7  uk_green.jpg
aae1c76f77c9b09f653e100e5f38b23344069ad0847d3d472457d41f12b3a3c2  uk_red_shrews.jpg
ce4818b7d7292728d9c917122e53ff63ebb0a2eb735f53301f6fe7343005dade  basel.jpg
28a4c3402174fe6f536e8bc9bef0ac2001e088e74d34a474181a04e9ebc5d010  ffm_vehicle.webm
dc3b74f4a5ae28ebd5d0ce744761b66c6a1e54720a0ea59b7fa197ecbe24ef15  ffm_vehicle.mp4 (local re-encode; may differ by ffmpeg version)
```

Licence links: CC BY-SA 3.0 https://creativecommons.org/licenses/by-sa/3.0/ , CC BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/ .

## Training data for the signal crop classifier (`streetsense/signal_cnn.py`)

Nothing below is committed except labels and the provenance manifest. `scripts/get_signal_data.sh` downloads the images to `$DATA` (default `/workspace/datasets`, outside git) and rebuilds the crops from `eval/signal_labels.csv.gz` (14,711 crops: source, file, box in source pixels, split, label W/S/C/O, and whether the label was manual or automatic). Downloaded 28 Oct 2026.

| Source | What we use | Licence | Notes |
|---|---|---|---|
| PTL dataset, Yu et al. "LYTNet" (https://github.com/samuelyu2002/PTL-Dataset) | 4,016 Shanghai street photos (4032×3024, two Google Drive zips, ~23 GB); 10,091 light crops | MIT (repo `LICENSE`) | Its labels are per-image crossing phases (red / green / countdown green / countdown blank / none), not boxes. Boxes come from our YOLO pass; crop labels are colour buckets plus eye review of 1,540 green crops. Its train/val/test split is kept. The full-res zips have broken zip64 offsets; `scripts/signal_data/ptl_zip.py` reads them. |
| Wikimedia Commons, "Category:Pedestrian signals by country" (depth ≤ 4) | 1,565 photos (1280 px thumbnails); 4,023 light crops | Per file: CC BY-SA 2.0–4.0 (1,203), CC BY 2.0–4.0 (184), CC0 (122), public domain (42), FAL (10), "Attribution" (3), "Copyrighted free use" (1) | Title, licence, author and page per file in `eval/signal_commons_manifest.csv`. 5 files with other licences were skipped at download. 6 titles that look like our held-out stills (aHash ≤ 20) are excluded (`eval/signal_commons_excluded.json`). Split 80/10/10 by file. |
| Our dev eval clips (`dc`, `eu_ped_a`, `eu_ped_b`, `night_ped`, `most11`, `us_ped`, `tirana`; sources above) | 597 tight crops at 2 fps | as listed above | Phase timeline + colour rules + eye check. Val = every fifth 10 s block. **Held-out clips and stills (`eu_ped_side`, `night_jp`, `most02/03/04`, `night_hand`, `beacon_*`, `night_square`, `tyumen`, `cdmx`, `mazatlan`, `tomaszow`, `samples/heldout/*`) are never used for training or validation.** |

Looked at and not used:
- Bosch Small Traffic Lights and DriveU DTLD: non-commercial research licences. Not usable for a competition entry.
- Ampel-Pilot (MIT): the image links in its SQL dump return 404.
- Mendeley "pedestrian traffic light" (809 images): web-scraped, unclear provenance.
- Roboflow Universe sets: need an account; licences vary per set.
- Zenodo ped-signal-data: timing data, no images.

## Commons signal videos (`eval/signal_video_clips.csv`, `eval/signal_video_labels.csv`)

Downloaded 29 Oct 2026 from Wikimedia Commons ("Category:Videos of traffic signals", its countdown subcategory, and a filetype:video search). Licence and author were read from the Commons API (`extmetadata`) that day. Videos live in `/workspace/datasets/commons_video` (not committed); `scripts/get_signal_videos.sh` re-fetches them and rebuilds the crops.

Boxes: our detector at 2 fps (YOLO11s OpenVINO, imgsz 960, zoom-out 0.5 when nothing found, plus 2×3 tiles). Labels: `scripts/signal_data/label_video_crops.py` (per-clip rules + HSV lit buckets; flashing beacons and vehicle heads → O), then every walk candidate was checked by eye on contact sheets and the rules tightened until the walk set was clean; 4 manual fixes in `eval/signal_video_fixes.json`. Split by clip: train clips = crossing_fr, toronto_diag, dus_r2g, faulty, wien, chelyabinsk, toronto_pxo; all others are test only.

| id | split | licence | author | crops W/S/C/O | what it shows | page |
|---|---|---|---|---|---|---|
| `wien` | train | Public domain | Priwo | 0/0/0/155 | Vienna, AT. Vehicle head cycle. | https://commons.wikimedia.org/wiki/File:Ampel-Wien-01.ogv |
| `china` | test | CC0 | XZise | 0/51/2/58 | China. Red man + red countdown digits, vehicle round lamps. | https://commons.wikimedia.org/wiki/File:Chinese_Signal.webm |
| `crossing_fr` | train | CC BY-SA 3.0 | Pleclown (talk) | 54/143/0/347 | France. Green-panel white walking man / red man, vehicle heads, night-ish dusk street. | https://commons.wikimedia.org/wiki/File:Crossing_light.ogv |
| `sanjose` | test | CC BY-SA 4.0 | Minh Nguyen | 0/0/0/9 | San José airport, US. Crosswalk, small heads. | https://commons.wikimedia.org/wiki/File:Crosswalk_at_San_Jos%C3%A9_International_Airport_Consolidated_Rental_Car_Facility_exit.webm |
| `toronto_diag` | train | CC0 | Raysonho @ Open Grid Scheduler / Grid Engine | 12/307/283/166 | Toronto, CA. Yonge–Dundas scramble: white walking person, orange hand + countdown digits, vehicle heads. | https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm |
| `dus_g2r` | test | Public domain | noebse | 13/7/0/56 | Düsseldorf, DE. Pedestrian green man → amber → red (3-lamp ped head), day. | https://commons.wikimedia.org/wiki/File:D%C3%BCsseldorfer_Ampel_Gr%C3%BCn_nach_Rot.ogv |
| `dus_r2g` | train | Public domain | noebse | 6/14/0/34 | Düsseldorf, DE. Same heads, red → green. | https://commons.wikimedia.org/wiki/File:D%C3%BCsseldorfer_Ampel_von_Rot_nach_Gr%C3%BCn.ogv |
| `faulty` | train | CC BY 2.0 | Franklin Rainier | 0/0/0/233 | Vehicle head with red countdown digits, faulty/flicker. | https://commons.wikimedia.org/wiki/File:Faulty_traffic_light_(1).webm |
| `kyiv_floor` | excluded | CC0 | Роман Рябенко | 0/0/0/0 | Kyiv, UA. In-ground floor lights. No boxes found; not labelled. | https://commons.wikimedia.org/wiki/File:Floor_traffic_lights_at_Pushkinska_Street_in_Kyiv.webm |
| `gau` | test | CC0 | ManuelB701 | 0/0/0/77 | Gau-Bischofsheim, DE. Vehicle head to red. | https://commons.wikimedia.org/wiki/File:Gau-Bischofsheim_-_Steigstra%C3%9Fe,_Ampelschaltung_auf_Rot.webm |
| `montreal` | test | CC BY-SA 4.0 | Thomas1313 | 0/0/0/9 | Montréal, CA. Intersection, small heads. | https://commons.wikimedia.org/wiki/File:Intersection_Pie-IX-Sherbrooke.webm |
| `tw_chaozhou` | test | CC BY-SA 4.0 | Tze Chiang Hao | 0/10/24/0 | Taiwan. Animated green man (Xiaolüren) + green countdown, then red man. | https://commons.wikimedia.org/wiki/File:June_2023_Xiaol%C3%BCren,_pedestrian_signal_of_Taiwan_in_Chaozhou_Township.webm |
| `tw_kaohsiung` | excluded | CC BY-SA 4.0 | Tze Chiang Hao | 0/0/0/0 | Taiwan. Combined pedestrian + bicycle heads with countdown. Not labelled (mixed heads). | https://commons.wikimedia.org/wiki/File:June_2023_pedestrian_and_bicycle_signal_in_Kaohsiung.webm |
| `tw_pingtung` | test | CC BY-SA 4.0 | Tze Chiang Hao | 14/8/15/0 | Taiwan. Dot-matrix green man + countdown (digits blank near end), red man. | https://commons.wikimedia.org/wiki/File:June_2023_pedestrian_signal_for_elderly_people_in_Pingtung.webm |
| `ottawa_temp` | excluded | CC0 | 137a | 0/0/0/0 | Ottawa, CA. Temporary vehicle signal. AV1, not decoded on the box; not labelled. | https://commons.wikimedia.org/wiki/File:Temporary_traffic_signal_at_Colonel_By_Drive.webm |
| `us_blinkred` | test | Public domain | SPUI | 0/0/0/17 | US. Vehicle signal blinking red (flashing). | https://commons.wikimedia.org/wiki/File:Traffic_signal_blinking_red_at_bottom.ogv |
| `chelyabinsk` | train | CC BY-SA 4.0 | Ural-66 | 0/0/0/128 | Chelyabinsk, RU. Vehicle head. | https://commons.wikimedia.org/wiki/File:%D0%90%D0%B2%D1%82%D0%BE%D0%BC%D0%BE%D0%B1%D0%B8%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9_%D1%81%D0%B2%D0%B5%D1%82%D0%BE%D1%84%D0%BE%D1%80_(%D0%A7%D0%B5%D0%BB%D1%8F%D0%B1%D0%B8%D0%BD%D1%81%D0%BA,_%D0%B2%D0%B8%D0%B4%D0%B5%D0%BE)_v001.webm |
| `pl_przejscie` | test | CC BY-SA 3.0 | shot by Jakub Sochacki, movie created by Julo | 0/2/0/5 | Poland. Pedestrian crossing, 1.7 s. | https://commons.wikimedia.org/wiki/File:Przejsciedlapieszych.ogv |
| `toronto_pxo` | train | CC BY-SA 4.0 | Qviri | 0/0/0/283 | Toronto, CA. PXO Type A crossing, flashing amber beacons (flashing, no ped head). | https://commons.wikimedia.org/wiki/File:Roncesvalles_Ave_crossing_at_Galley_Ave,_PXO_Type_A_lights_flashing,_July_2026.webm |
| `tw_flicker` | test | CC BY 4.0 | Fumikas Sagisavas | 0/0/0/28 | Taiwan. Flashing yellow / slow-light vehicle signal (flashing). | https://commons.wikimedia.org/wiki/File:The_traffic_signal_light_of_the_flickering_yellow_light_and_the_slow_light.webm |

Excluded as duplicates of clips we already use: `Pedestrian crossing light washington dc.ogv` (= `dc`), the CDMX Eje Central clip (= `cdmx`), the Most, Tirana, NZ and FFM clips. Skipped: Katella Anaheim (500 MB), Tbilisi and Traffic Light Tree (art installation, >100 MB), a 1950s b/w public-information film.

Label noise seen on the sheets: a few partly visible French walk figures are labelled O; vehicle red lamps in `crossing_fr` and `china` are labelled S rather than O; Toronto hand vs hand+digits is split by box shape (wide → C), so some S/C swaps. No non-walk crop is labelled W on the review sheets.

## Not obtained (licence unconfirmed, download blocked)

- **PTLD (Brazil, PUCRS)**: https://tinyurl.com/y9z7xodw → Google Drive folder `0B2MY7T7S8OmJVVlCTW1jYWxqUVE`. On 29 Oct 2026 Drive returned a sign-in page (HTTP 401 for gdown; legacy link without resource key). Licence: **unconfirmed**. Nothing downloaded.
- **PTL-Crosswalk (Brazil/France)**: https://github.com/ronaldosm/PedestrianTrafficLightsAndCrosswalkDetection → OneDrive folder. On 29 Oct 2026 the OneDrive share and its API both returned "authentication required". Licence: **unconfirmed** (no LICENSE in the repo). Nothing downloaded.
- Retry with `scripts/get_ptld_ptlcw.sh` (re-checked 29 Oct 2026 14:25 UK time: still 401/403). If either is obtained later: store under `/workspace/datasets/ptld` / `ptlcw`, never commit images or crops, and record the licence here as unconfirmed until the authors reply.
