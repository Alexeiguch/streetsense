import json, os
O = "/workspace/sr_logs/rf"
def own(V, T, R, F):
    p = f"{O}/own_{V}_{T}_{R}_y{F}.json"
    if not os.path.exists(p): return None
    d = json.load(open(p)); s = d["summary"]
    fwids = [r["id"] for r in d["rows"] if r["label"] != "walk" and r["local"] == "walk"]
    dense = sum(v["walk"] for v in d["dense_info"].values())
    return s, fwids, dense
def ptld(V, T, R, F):
    p = f"{O}/ptld_{V}_{T}_{R}_y{F}.json"
    if not os.path.exists(p): return None
    return json.load(open(p))
dn = {"default": "default", "pedsig_ptld": "pedsig"}
print("### PTLD test, 478 full images (217 walk, 261 non-walk)\n")
print("| Detector + reader | Tiles | Flag | Walk heads found | Walk read | **False walks** | Crop walk read (of 240) | Crop false walks (of 284) | Median / p90 ms |")
print("|---|---|---|---|---|---|---|---|---|")
for V in dn:
    for T in ("off", "2x3"):
        for R in ("v1", "v5"):
            for F in "01":
                d = ptld(V, T, R, F)
                if d is None: print(f"| {dn[V]} + {R} | {T} | {'on' if F=='1' else 'off'} | not run | | | | | |"); continue
                ip, cr = d["images_pipeline"], d["crops_reader"]
                fw = sum(v["said_walk"] for k, v in ip.items() if k != "walk"); nn = sum(v["n"] for k, v in ip.items() if k != "walk")
                cfw = sum(v["said_walk"] for k, v in cr.items() if k != "walk"); cn = sum(v["n"] for k, v in cr.items() if k != "walk")
                print(f"| {dn[V]} + {R} | {T} | {'on' if F=='1' else 'off'} | {d['walk_head_detected'][0]} | {ip['walk']['said_walk']} | **{fw}/{nn}** | {cr['walk']['said_walk']} | {cfw}/{cn} | {d['frame_ms']['median']} / {d['frame_ms']['p90']} |")
print("\n### Own clips (36 walk, 72 non-walk labelled frames; dev 22 / held-out 14 walks)\n")
print("| Detector + reader | Tiles | Flag | Walks read | **False walks** | Go (cached Muse) | **False go** | Dev walks | Held-out walks | Dense dc non-walk walk decisions |")
print("|---|---|---|---|---|---|---|---|---|---|")
for V in dn:
    for T in ("off", "2x3"):
        for R in ("v1", "v5"):
            for F in "01":
                x = own(V, T, R, F)
                if x is None: print(f"| {dn[V]} + {R} | {T} | {'on' if F=='1' else 'off'} | not run |||||||"); continue
                s, ids, dense = x; a, dv, h = s["ALL"], s["dev/ALL"], s["heldout/ALL"]
                print(f"| {dn[V]} + {R} | {T} | {'on' if F=='1' else 'off'} | {a['reader_walk']}/{a['walk_frames']} | **{a['false_walk']}/{a['nonwalk_frames']}**{' (' + ', '.join(ids) + ')' if ids else ''} | {a['go']}/{a['muse_cached']} | **{a['false_go']}** | {dv['reader_walk']}/{dv['walk_frames']} | {h['reader_walk']}/{h['walk_frames']} | {dense} |")
