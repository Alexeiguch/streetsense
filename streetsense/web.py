"""Phone-camera demo. Open http://<box>:8000 on a phone (HTTPS or localhost needed for camera access).

The page grabs camera frames, POSTs them as JPEG to /frame, draws the returned boxes,
and speaks alerts with the browser's speechSynthesis. "Can I cross?" calls Muse Spark
with local+Muse fusion. Optional STREETSENSE_DEMO_TOKEN gates billable /ask.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, make_response, request

from .config import Config
from .pipeline import Pipeline

MAX_FRAME_BYTES = 4_000_000  # 4 MB
SESSION_TTL_S = 600.0
ASK_RATE_LIMIT = 6  # per session per minute
ASK_MIN_INTERVAL_S = 3.0

PAGE = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>StreetSense</title><style>
body{font-family:system-ui;background:#000;color:#fff;margin:0;text-align:center}
#v{display:none}#c{width:100%;max-width:720px}
#alert{font-size:2rem;font-weight:700;min-height:2.6rem;padding:.4rem}
button{font-size:1.4rem;padding:1rem 1.6rem;margin:.4rem;border-radius:1rem;border:0}
#go{background:#1db954}#stop{background:#c0392b}#ask{background:#ffd400}#info{font-size:.9rem;color:#aaa}
</style></head><body>
<div id=alert aria-live=assertive>Tap Start</div>
<button id=go>Start</button><button id=stop>Stop</button>
<button id=ask>Can I cross?</button><button id=mic style="background:#6cf">Hold to ask (voice)</button>
<video id=v playsinline muted></video><canvas id=c></canvas><div id=info></div>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),x=c.getContext('2d');
const al=document.getElementById('alert'),info=document.getElementById('info');
let running=false, starting=false, stream=null, askSeq=0;
function say(t){if(!t)return;al.textContent=t;const u=new SpeechSynthesisUtterance(t);u.rate=1.15;speechSynthesis.cancel();speechSynthesis.speak(u);}
function stopCamera(){
  running=false; starting=false;
  if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;}
  v.srcObject=null; say('StreetSense stopped');
}
async function loop(){ if(!running)return;
  try{
    if(!v.videoWidth){say('camera not ready'); setTimeout(loop,200); return;}
    const w=640,h=Math.round(v.videoHeight*640/v.videoWidth)||480; c.width=w;c.height=h; x.drawImage(v,0,0,w,h);
    const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.7));
    if(!blob||blob.size<100){say('camera covered or no frame'); setTimeout(loop,200); return;}
    const t0=performance.now();
    const r=await fetch('/frame',{method:'POST',body:blob,credentials:'same-origin'});
    if(!r.ok){say('network error '+r.status); info.textContent='frame HTTP '+r.status; setTimeout(loop,300); return;}
    const j=await r.json();
    x.lineWidth=3;x.font='16px system-ui';
    for(const d of (j.detections||[])){x.strokeStyle=d.color;x.strokeRect(d.box[0],d.box[1],d.box[2]-d.box[0],d.box[3]-d.box[1]);
      x.fillStyle=d.color;x.fillText(d.text,d.box[0]+2,d.box[1]+16);}
    if(j.spoken)say(j.spoken.text);
    if(j.camera_health&&j.camera_health.issue)say(j.camera_health.message||('camera '+j.camera_health.issue));
    info.textContent=`round trip ${(performance.now()-t0).toFixed(0)} ms, server ${j.server_ms.toFixed(0)} ms`;
  }catch(e){say('network error'); info.textContent='error '+e}
  if(running) setTimeout(loop,50);
}
document.getElementById('go').onclick=async()=>{
  if(running||starting)return;
  starting=true;
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
    v.srcObject=stream;await v.play();running=true;say('StreetSense on');loop();
  }catch(e){say('camera permission failed'); info.textContent=String(e);}
  finally{starting=false;}
};
document.getElementById('stop').onclick=()=>stopCamera();
document.getElementById('ask').onclick=async()=>{
  const my=++askSeq;
  const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.85));
  al.textContent='asking Muse Spark...';
  const headers={}; const tok=localStorage.getItem('streetsense_token'); if(tok)headers['X-Demo-Token']=tok;
  headers['X-Request-Id']=String(my);
  try{
    const r=await fetch('/ask',{method:'POST',body:blob,credentials:'same-origin',headers});
    const j=await r.json();
    if(my!==askSeq)return; // drop out-of-order
    say(j.spoken||j.error||'no answer');
  }catch(e){if(my===askSeq)say('ask failed');}
};

document.getElementById('mic').onpointerdown=async(ev)=>{
  ev.preventDefault();
  if(!window.MediaRecorder){say('voice unsupported');return;}
  try{
    const s=await navigator.mediaDevices.getUserMedia({audio:true,video:false});
    const rec=new MediaRecorder(s); const chunks=[];
    rec.ondataavailable=e=>chunks.push(e.data);
    rec.onstop=async()=>{
      s.getTracks().forEach(t=>t.stop());
      const blob=new Blob(chunks,{type:'audio/webm'});
      al.textContent='transcribing...';
      const r=await fetch('/transcribe',{method:'POST',body:blob,credentials:'same-origin'});
      const j=await r.json();
      const q=(j.transcript||'').trim()||'Can I cross?';
      say('heard: '+q);
      const my=++askSeq;
      const frameBlob=await new Promise(res=>c.toBlob(res,'image/jpeg',0.85));
      const headers={'X-Request-Id':String(my)};
      const tok=localStorage.getItem('streetsense_token'); if(tok)headers['X-Demo-Token']=tok;
      const ar=await fetch('/ask?q='+encodeURIComponent(q),{method:'POST',body:frameBlob,credentials:'same-origin',headers});
      const aj=await ar.json();
      if(my!==askSeq)return;
      say(aj.spoken||aj.error||'no answer');
    };
    rec.start(); al.textContent='listening... release to send';
    const up=()=>{rec.stop(); window.removeEventListener('pointerup',up);};
    window.addEventListener('pointerup',up);
  }catch(e){say('mic error'); info.textContent=String(e);}
};
</script></body></html>"""

COL = {"red": "#ff3030", "green": "#20e020", "yellow": "#ffc000", "walk": "#ffffff", "unclear": "#ff80ff"}


@dataclass
class SessionState:
    pipe: Pipeline
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = field(default_factory=time.monotonic)
    ask_times: list[float] = field(default_factory=list)
    ask_inflight: bool = False
    camera_gen: int = 0


def _demo_token() -> str | None:
    return os.environ.get("STREETSENSE_DEMO_TOKEN") or None


def camera_health(img: np.ndarray, last_frame_ts: float | None = None) -> dict | None:
    """Cheap heuristics: covered / blurred / pointed too high / stale."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sky_ratio = float((gray[: max(1, h // 3)] > 200).mean()) if h > 10 else 0.0
    if mean < 18:
        return {"issue": "covered", "message": "camera covered or too dark"}
    if var < 12:
        return {"issue": "blurred", "message": "camera image is blurred"}
    if sky_ratio > 0.85 and mean > 160:
        return {"issue": "pointed_high", "message": "camera pointed too high"}
    if last_frame_ts is not None and (time.monotonic() - last_frame_ts) > 3.0:
        return {"issue": "stale", "message": "no fresh camera frames"}
    return None


def create_app(no_surface: bool = False, backend: str | None = None, weights: str | None = None,
               surface_backend: str | None = None) -> Flask:
    app = Flask(__name__)
    cfg = Config()
    if backend is not None:
        cfg.detector.backend = backend
        if surface_backend is None:
            cfg.surface.backend = surface_backend or backend
    if surface_backend is not None:
        cfg.surface.backend = surface_backend
    if weights is not None:
        cfg.detector.weights = weights

    # Shared stateless-ish config; per-session Pipelines hold tracker/votes/cooldowns.
    sessions: dict[str, SessionState] = {}
    sessions_lock = threading.Lock()

    def _mk_pipe() -> Pipeline:
        return Pipeline(cfg, use_surface=not no_surface)

    def _purge(now: float) -> None:
        dead = [k for k, s in sessions.items() if now - s.last_seen > SESSION_TTL_S]
        for k in dead:
            sessions.pop(k, None)

    def get_session() -> tuple[str, SessionState]:
        now = time.monotonic()
        sid = request.cookies.get("ss_sid") or request.headers.get("X-Session-Id")
        with sessions_lock:
            _purge(now)
            if sid and sid in sessions:
                s = sessions[sid]
                s.last_seen = now
                return sid, s
            sid = secrets.token_hex(16)
            s = SessionState(pipe=_mk_pipe(), last_seen=now)
            sessions[sid] = s
            return sid, s

    def decode(body: bytes) -> np.ndarray | None:
        if body is None or len(body) == 0:
            return None
        if len(body) > MAX_FRAME_BYTES:
            return None
        try:
            arr = np.frombuffer(body, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
        return img

    def attach_sid(resp, sid: str):
        resp.set_cookie("ss_sid", sid, max_age=int(SESSION_TTL_S), httponly=True, samesite="Lax")
        return resp

    def check_ask_auth() -> tuple[bool, str | None]:
        expected = _demo_token()
        if not expected:
            return True, None
        got = request.headers.get("X-Demo-Token") or request.args.get("token")
        if not got or not secrets.compare_digest(got, expected):
            return False, "unauthorized"
        return True, None

    @app.get("/")
    def index():
        return PAGE

    @app.post("/frame")
    def frame():
        body = request.get_data()
        if not body:
            return jsonify(error="empty body"), 400
        if len(body) > MAX_FRAME_BYTES:
            return jsonify(error="frame too large"), 400
        img = decode(body)
        if img is None:
            return jsonify(error="bad image"), 400
        sid, sess = get_session()
        # Reset temporal state when client signals camera restart.
        if request.headers.get("X-Camera-Restart") == "1":
            with sess.lock:
                sess.pipe = _mk_pipe()
                sess.camera_gen += 1
        t0 = time.perf_counter()
        health = camera_health(img)
        with sess.lock:
            out = sess.pipe.process(img, annotate=False)
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
        resp = make_response(jsonify(
            detections=dets, spoken=spoken, server_ms=(time.perf_counter() - t0) * 1000,
            underfoot=out.surface.underfoot if out.surface else None,
            camera_health=health, session=sid[:8],
        ))
        return attach_sid(resp, sid)

    @app.post("/ask")
    def ask():
        ok, err = check_ask_auth()
        if not ok:
            return jsonify(error=err), 401
        body = request.get_data()
        if not body or len(body) > MAX_FRAME_BYTES:
            return jsonify(error="bad image"), 400
        img = decode(body)
        if img is None:
            return jsonify(error="bad image"), 400
        sid, sess = get_session()
        now = time.monotonic()
        with sess.lock:
            sess.ask_times = [t for t in sess.ask_times if now - t < 60.0]
            if len(sess.ask_times) >= ASK_RATE_LIMIT:
                return jsonify(error="rate limited"), 429
            if sess.ask_times and now - sess.ask_times[-1] < ASK_MIN_INTERVAL_S:
                return jsonify(error="too soon"), 429
            if sess.ask_inflight:
                return jsonify(error="ask in flight"), 409
            sess.ask_inflight = True
            sess.ask_times.append(now)
        capture_ts = now
        try:
            from .meta_api import (
                SAFE_CROSSING_FALLBACK,
                fuse_crossing_decision,
                render_crossing_speech,
                validate_crossing_reply,
            )
            from .meta_usage import get_ledger
            from .rules import select_crossing_signal

            q = request.args.get("q", "Is it safe to cross the street right now?")
            with sess.lock:
                out = sess.pipe.process(img, annotate=False)
            muse_raw = sess.pipe.vlm.ask_sync(img, q)
            validated = validate_crossing_reply(
                muse_raw if isinstance(muse_raw, dict) and "error" not in muse_raw else None,
                capture_ts=capture_ts,
            )
            H, W = img.shape[:2]
            sel = select_crossing_signal(out.detections, W, H, out.crosswalk)
            local = sel.decision if sel is not None else "unknown"
            spoken = render_crossing_speech(validated, local_state=local, now=time.monotonic())
            fused = fuse_crossing_decision(local, validated.get("data") if validated.get("ok") else None)
            if not validated.get("ok"):
                spoken = SAFE_CROSSING_FALLBACK
            payload = {
                **(muse_raw if isinstance(muse_raw, dict) else {}),
                "local_light": local,
                "fuse": fused,
                "spoken": spoken,
                "validated": bool(validated.get("ok")),
                "capture_ts": capture_ts,
                "request_id": request.headers.get("X-Request-Id"),
                "meta_usage": get_ledger().summary(),
            }
            resp = make_response(jsonify(payload))
            return attach_sid(resp, sid)
        finally:
            with sess.lock:
                sess.ask_inflight = False

    @app.post("/transcribe")
    def transcribe():
        """Optional Muse Voice Transcribe. Browser may send webm; we remux to WAV when possible."""
        import subprocess
        import tempfile
        from .meta_api import MuseVoiceTranscribe
        raw = request.get_data()
        if not raw:
            return jsonify(error="empty audio"), 400
        if len(raw) > MAX_FRAME_BYTES:
            return jsonify(error="audio too large"), 400
        vt = MuseVoiceTranscribe()
        if not vt.enabled:
            return jsonify(error="voice disabled: set MODEL_API_KEY"), 503
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "in.webm"
            wav = Path(td) / "out.wav"
            src.write_bytes(raw)
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "24000",
                     "-c:a", "pcm_s16le", "-map_metadata", "-1", str(wav)],
                    check=True, capture_output=True, timeout=30,
                )
            except Exception as e:
                return jsonify(error=f"ffmpeg convert failed: {type(e).__name__}"), 400
            return jsonify(vt.transcribe_wav(str(wav)))

    # Expose for tests
    app.extensions["streetsense_sessions"] = sessions
    app.extensions["streetsense_mk_pipe"] = _mk_pipe
    return app
