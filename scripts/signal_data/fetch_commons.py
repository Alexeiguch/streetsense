#!/usr/bin/env python3
"""Download the Wikimedia Commons files listed in eval/signal_commons_manifest.csv
(1280 px thumbnails; per-file licences in that manifest, all CC BY*/CC BY-SA*/CC0/PD/FAL)."""
import csv, hashlib, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/commons/img")
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "streetsense-eval/1.0 (research; pedestrian signal classifier)"}
titles = [r["title"] for r in csv.DictReader(open(ROOT / "eval/signal_commons_manifest.csv"))]
for i in range(0, len(titles), 50):
    q = dict(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo", iiprop="url",
             iiurlwidth=1280, format="json")
    req = urllib.request.Request("https://commons.wikimedia.org/w/api.php",
                                 data=urllib.parse.urlencode(q).encode(), headers=UA)
    for p in json.load(urllib.request.urlopen(req))["query"]["pages"].values():
        ii = (p.get("imageinfo") or [{}])[0]
        url = ii.get("thumburl") or ii.get("url")
        fn = OUT / (hashlib.sha1(p["title"].encode()).hexdigest()[:16] + ".jpg")
        if not url or fn.exists():
            continue
        for k in range(4):
            try:
                fn.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
                break
            except Exception:
                time.sleep(3 * (k + 1))
print("commons files in", OUT, len(list(OUT.glob("*.jpg"))))
