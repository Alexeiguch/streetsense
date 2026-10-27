"""Phone-camera demo. Open http://<box>:8000 on a phone (HTTPS or localhost needed for camera access).

The page grabs camera frames, POSTs them as JPEG to /frame, draws the returned boxes,
and speaks alerts with the browser's speechSynthesis. "Can I cross?" calls Muse Spark.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

from .config import Config
from .pipeline import Pipeline

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>StreetSense</title><style>
body{font-family:system-ui;background:#000;color:#fff;margin:0;text-align:center}
#v{display:none}#c{width:100%;max-width:720px}
#alert{font-size:2rem;font-weight:700;min-height:2.6rem;padding:.4rem}
button{font-size:1.4rem;padding:1rem 1.6rem;margin:.4rem;border-radius:1rem;border:0}
#go{background:#1db954}#ask{background:#ffd400}#info{font-size:.9rem;color:#aaa}
</style></head><body>
<div id=alert aria-live=assertive>Tap Start</div>
<button id=go>Start</button><button id=ask>Can I cross?</button>
<video id=v playsinline muted></video><canvas id=c></canvas><div id=info></div>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),x=c.getContext('2d');
const al=document.getElementById('alert'),info=document.getElementById('info');
let running=false;
function say(t){al.textContent=t;const u=new SpeechSynthesisUtterance(t);u.rate=1.15;speechSynthesis.cancel();speechSynthesis.speak(u);}
async function loop(){ if(!running)return;
  const w=640,h=Math.round(v.videoHeight*640/v.videoWidth)||480; c.width=w;c.height=h; x.drawImage(v,0,0,w,h);
  const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.7));
  const t0=performance.now();
  try{const r=await fetch('/frame',{method:'POST',body:blob});const j=await r.json();
    x.lineWidth=3;x.font='16px system-ui';
    for(const d of j.detections){x.strokeStyle=d.color;x.strokeRect(d.box[0],d.box[1],d.box[2]-d.box[0],d.box[3]-d.box[1]);
      x.fillStyle=d.color;x.fillText(d.text,d.box[0]+2,d.box[1]+16);}
    if(j.spoken)say(j.spoken.text);
    info.textContent=`round trip ${(performance.now()-t0).toFixed(0)} ms, server ${j.server_ms.toFixed(0)} ms`;
  }catch(e){info.textContent='error '+e}
  setTimeout(loop,50);}
document.getElementById('go').onclick=async()=>{
  const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
  v.srcObject=s;await v.play();running=true;say('StreetSense on');loop();};
document.getElementById('ask').onclick=async()=>{
  const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.85));
  al.textContent='asking Muse Spark...';
  const r=await fetch('/ask',{method:'POST',body:blob});const j=await r.json();
  say(j.summary||j.error||'no answer');};
</script></body></html>"""

COL = {"red": "#ff3030", "green": "#20e020", "yellow": "#ffc000", "walk": "#ffffff"}


def create_app(no_surface: bool = False, backend: str | None = None, weights: str | None = None,
               surface_backend: str | None = None) -> Flask:
    app = Flask(__name__)
    cfg = Config()
    if backend is not None:
        cfg.detector.backend = backend
        if surface_backend is None:
            cfg.surface.backend = backend
    if surface_backend is not None:
        cfg.surface.backend = surface_backend
    if weights is not None:
        cfg.detector.weights = weights
    pipe = Pipeline(cfg, use_surface=not no_surface)
    lock = threading.Lock()

    def decode(body: bytes) -> np.ndarray | None:
        return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)

    @app.get("/")
    def index():
        return PAGE

    @app.post("/frame")
    def frame():
        img = decode(request.get_data())
        if img is None:
            return jsonify(error="bad image"), 400
        t0 = time.perf_counter()
        with lock:
            out = pipe.process(img, annotate=False)
        dets = []
        for d in out.detections:
            text = d.label
            if d.label == "traffic light":
                text = f"{d.attrs.get('signal_kind')} {d.attrs.get('light_state')}"
            elif d.attrs.get("distance_m") is not None:
                text += f" {d.attrs['distance_m']:.1f}m"
            dets.append({"box": [d.box.x1, d.box.y1, d.box.x2, d.box.y2], "text": text,
                         "color": COL.get(d.attrs.get("light_state"), "#00c8ff")})
        if out.crosswalk and out.crosswalk.found:
            b = out.crosswalk.box
            dets.append({"box": [b.x1, b.y1, b.x2, b.y2], "text": "crossing", "color": "#ffffff"})
        spoken = {"text": out.spoken.text, "priority": out.spoken.priority.name} if out.spoken else None
        return jsonify(detections=dets, spoken=spoken, server_ms=(time.perf_counter() - t0) * 1000,
                       underfoot=out.surface.underfoot if out.surface else None)

    @app.post("/ask")
    def ask():
        img = decode(request.get_data())
        if img is None:
            return jsonify(error="bad image"), 400
        return jsonify(pipe.vlm.ask_sync(img, request.args.get("q", "Is it safe to cross the street right now?")))

    return app
