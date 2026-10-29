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
<button id=guide style="background:#c9f">Guide me (Wonder)</button><button id=gm style="background:#ccc">Glasses mode: off</button>
<video id=v playsinline muted></video><canvas id=c></canvas><div id=info></div>
<script>
const v=document.getElementById('v'),c=document.getElementById('c'),x=c.getContext('2d');
const al=document.getElementById('alert'),info=document.getElementById('info');
let running=false, starting=false, stream=null, askSeq=0;
const qs=new URLSearchParams(location.search);
// Glasses mode mirrors a Ray-Ban toolkit stream: portrait, toolkit size and fps, short cues + earcons.
const GL_SIZES={low:[360,640],medium:[504,896],high:[720,1280]};
let glasses=qs.get('mode')==='glasses';
const glRes=GL_SIZES[qs.get('res')]?qs.get('res'):'medium';
const glFps=[2,7,15,24,30].includes(+qs.get('fps'))?+qs.get('fps'):15;
const EARCONS=__EARCONS__;
let actx=null;
function earcon(name){const e=EARCONS[name]; if(!e)return;
  try{actx=actx||new (window.AudioContext||window.webkitAudioContext)();
    for(let i=0;i<e.repeat;i++){const o=actx.createOscillator(),g=actx.createGain();
      const t=actx.currentTime+i*(e.ms+60)/1000; o.frequency.value=e.freq_hz; o.connect(g); g.connect(actx.destination);
      g.gain.setValueAtTime(0.25,t); g.gain.exponentialRampToValueAtTime(0.001,t+e.ms/1000); o.start(t); o.stop(t+e.ms/1000);}
  }catch(err){}
}
function say(t,rate){if(!t)return;al.textContent=t;const u=new SpeechSynthesisUtterance(t);u.rate=rate||1.15;speechSynthesis.cancel();speechSynthesis.speak(u);}
function setGm(){document.getElementById('gm').textContent='Glasses mode: '+(glasses?glRes+' '+glFps+' fps':'off');}
function stopCamera(){
  running=false; starting=false;
  if(stream){stream.getTracks().forEach(t=>t.stop()); stream=null;}
  v.srcObject=null; say('StreetSense stopped');
}
async function loop(){ if(!running)return;
  const tLoop=performance.now();
  try{
    if(!v.videoWidth){say('camera not ready'); setTimeout(loop,200); return;}
    let w=640,h=Math.round(v.videoHeight*640/v.videoWidth)||480;
    if(glasses){[w,h]=GL_SIZES[glRes]; c.width=w;c.height=h;
      const tr=w/h, sr=v.videoWidth/v.videoHeight; let sw=v.videoWidth, sh=v.videoHeight, sx=0, sy=0;
      if(sr>tr){sw=Math.round(sh*tr); sx=Math.round((v.videoWidth-sw)/2);} else {sh=Math.round(sw/tr); sy=Math.round((v.videoHeight-sh)/2);}
      x.drawImage(v,sx,sy,sw,sh,0,0,w,h);
    } else {c.width=w;c.height=h; x.drawImage(v,0,0,w,h);}
    const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',glasses?0.6:0.7));
    if(!blob||blob.size<100){say('camera covered or no frame'); setTimeout(loop,200); return;}
    const t0=performance.now();
    const r=await fetch(glasses?'/frame?mode=glasses':'/frame',{method:'POST',body:blob,credentials:'same-origin'});
    if(!r.ok){say('network error '+r.status); info.textContent='frame HTTP '+r.status; setTimeout(loop,300); return;}
    const j=await r.json();
    x.lineWidth=3;x.font='16px system-ui';
    for(const d of (j.detections||[])){x.strokeStyle=d.color;x.strokeRect(d.box[0],d.box[1],d.box[2]-d.box[0],d.box[3]-d.box[1]);
      x.fillStyle=d.color;x.fillText(d.text,d.box[0]+2,d.box[1]+16);}
    if(glasses){ if(j.glasses){earcon(j.glasses.earcon); say(j.glasses.text,1.3);} }
    else if(j.spoken)say(j.spoken.text);
    if(j.camera_health&&j.camera_health.issue)say(j.camera_health.message||('camera '+j.camera_health.issue));
    info.textContent=`round trip ${(performance.now()-t0).toFixed(0)} ms, server ${j.server_ms.toFixed(0)} ms`;
  }catch(e){say('network error'); info.textContent='error '+e}
  if(running) setTimeout(loop,glasses?Math.max(0,1000/glFps-(performance.now()-tLoop)):50);
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

document.getElementById('gm').onclick=()=>{glasses=!glasses; setGm(); say(glasses?'glasses mode':'phone mode');};
setGm();
document.getElementById('guide').onclick=async()=>{
  const my=++askSeq;
  const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.85));
  const q=prompt('Ask the guide','How do I get to the crossing?')||'What is around me?';
  al.textContent='asking Muse Wonder...';
  const headers={'X-Request-Id':String(my)}; const tok=localStorage.getItem('streetsense_token'); if(tok)headers['X-Demo-Token']=tok;
  try{
    const r=await fetch('/guide?q='+encodeURIComponent(q),{method:'POST',body:blob,credentials:'same-origin',headers});
    const j=await r.json(); if(my!==askSeq)return;
    say(j.speech||j.error||'no answer');
  }catch(e){if(my===askSeq)say('guide failed');}
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
            s = sessions.pop(k, None)
            if s is not None and getattr(s.pipe, "sam_fallback", None) is not None:
                print(f"[sam-fallback] session end: {s.pipe.sam_fallback_summary()}")

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
        import json as _json

        from .glasses import EARCONS

        return PAGE.replace("__EARCONS__", _json.dumps(EARCONS))

    @app.get("/glasses")
    def glasses_page():
        from flask import redirect

        q = request.query_string.decode()
        return redirect("/?mode=glasses" + ("&" + q if q else ""))

    @app.get("/health")
    def health():
        # Same shape the OpenClaw gateway returns; VisionClaw checks it before asking.
        return jsonify(ok=True, status="live")

    def admit_billable(sess: SessionState, now: float):
        """Rate limit + single flight for billable routes. Returns an error response or None."""
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
        return None

    def release_billable(sess: SessionState) -> None:
        with sess.lock:
            sess.ask_inflight = False

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
        extra = {}
        if request.args.get("mode") == "glasses":
            from .glasses import glasses_cue

            extra["glasses"] = glasses_cue(out.spoken)
        resp = make_response(jsonify(
            detections=dets, spoken=spoken, server_ms=(time.perf_counter() - t0) * 1000,
            underfoot=out.surface.underfoot if out.surface else None,
            camera_health=health, session=sid[:8], **extra,
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
        denied = admit_billable(sess, now)
        if denied is not None:
            return denied
        capture_ts = now
        try:
            from .answers import crossing_answer
            from .meta_usage import get_ledger

            q = request.args.get("q", "Is it safe to cross the street right now?")
            with sess.lock:
                out = sess.pipe.process(img, annotate=False)
            ans = crossing_answer(sess.pipe, img, q, out=out, capture_ts=capture_ts)
            payload = {
                **ans["muse_raw"],
                "local_light": ans["local_light"],
                "fuse": ans["fuse"],
                "spoken": ans["spoken"],
                "validated": ans["validated"],
                "capture_ts": capture_ts,
                "request_id": request.headers.get("X-Request-Id"),
                "meta_usage": get_ledger().summary(),
            }
            resp = make_response(jsonify(payload))
            return attach_sid(resp, sid)
        finally:
            release_billable(sess)

    @app.post("/guide")
    def guide():
        """Opt-in Muse Wonder guide (STREETSENSE_WONDER=1). Never authorises crossing."""
        from .config import WonderConfig

        wcfg = WonderConfig.from_env()
        if not wcfg.enabled:
            return jsonify(error="Muse Wonder guide is off: set STREETSENSE_WONDER=1"), 404
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
        denied = admit_billable(sess, time.monotonic())
        if denied is not None:
            return denied
        try:
            from .meta_usage import get_ledger
            from .wonder_agent import GuideTools, WonderGuide

            with sess.lock:
                out = sess.pipe.process(img, annotate=False)
            q = request.args.get("q", "What is around me?")[:300]
            res = make_guide(wcfg).run(q, img, GuideTools.from_pipeline(sess.pipe, img, out))
            resp = make_response(jsonify({**res.to_dict(), "request_id": request.headers.get("X-Request-Id"),
                                          "meta_usage": get_ledger().summary()}))
            return attach_sid(resp, sid)
        finally:
            release_billable(sess)

    @app.post("/v1/chat/completions")
    def chat_completions():
        """OpenClaw-compatible gateway endpoint so the cookbook VisionClaw app can point at StreetSense.

        Request: {model, messages:[{role:user, content:[{type:text},{type:image_url,image_url:{url:data:...}}]}], user}
        Auth: Authorization: Bearer <STREETSENSE_GATEWAY_TOKEN>. Off unless that token is set.
        """
        return gateway_reply()

    def gateway_reply():
        import hmac

        from .answers import crossing_answer
        from .wonder_agent import is_crossing_question, sanitize_guidance

        token = os.environ.get("STREETSENSE_GATEWAY_TOKEN")
        if not token:
            return jsonify(error={"message": "gateway disabled: set STREETSENSE_GATEWAY_TOKEN"}), 503
        auth = request.headers.get("Authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {token}".encode()):
            return jsonify(error={"message": "unauthorized"}), 401
        if (request.content_length or 0) > MAX_FRAME_BYTES * 2:
            return jsonify(error={"message": "request too large"}), 413
        body = request.get_json(silent=True) or {}
        if body.get("stream"):
            return jsonify(error={"message": "stream=true not supported"}), 400
        q, img = parse_chat_turn(body)
        if img is None:
            return jsonify(error={"message": "one image_url data URI is required"}), 400
        user = str(body.get("user") or "gateway")[:64]
        sid = "gw:" + user
        now = time.monotonic()
        with sessions_lock:
            _purge(now)
            sess = sessions.get(sid)
            if sess is None:
                sess = SessionState(pipe=_mk_pipe())
                sessions[sid] = sess
            sess.last_seen = now
        denied = admit_billable(sess, now)
        if denied is not None:
            return denied
        try:
            with sess.lock:
                if hasattr(sess.pipe, "smoother"):
                    # Photos arrive seconds apart: never let votes from earlier photos
                    # add up to a confirmed walk. Each photo is read on its own.
                    from .traffic_light import LightStateSmoother

                    sess.pipe.smoother = LightStateSmoother()
                out = sess.pipe.process(img, annotate=False)
            from .config import WonderConfig

            wcfg = WonderConfig.from_env()
            meta = {"route": None}
            if is_crossing_question(q):
                ans = crossing_answer(sess.pipe, img, q or "Is it safe to cross the street right now?", out=out)
                text = ans["spoken"]
                meta.update(route="crossing", local_light=ans["local_light"], fuse=ans["fuse"])
            elif wcfg.enabled:
                from .wonder_agent import GuideTools

                res = make_guide(wcfg).run(q, img, GuideTools.from_pipeline(sess.pipe, img, out))
                text = res.speech
                meta.update(route="wonder", turns=res.turns, cost_usd=res.cost_usd)
            else:
                raw = sess.pipe.vlm.ask_sync(img, q or "What is in front of me?")
                summary = raw.get("summary", "") if isinstance(raw, dict) else ""
                text, dropped = sanitize_guidance(summary)
                text = text or "I'm not sure. Please ask again."
                meta.update(route="spark", dropped=len(dropped))
            return jsonify({
                "id": "chatcmpl-ss-" + secrets.token_hex(6),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model") or "streetsense",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "streetsense": meta,
            })
        finally:
            release_billable(sess)

    def parse_chat_turn(body: dict):
        """Last user message -> (question text, BGR image or None)."""
        import base64

        msgs = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(msgs, list):
            return "", None
        user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        if not user_msgs:
            return "", None
        content = user_msgs[-1].get("content")
        if isinstance(content, str):
            return content.strip()[:300], None
        texts, img = [], None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif part.get("type") == "image_url" and img is None:
                url = (part.get("image_url") or {}).get("url", "") if isinstance(part.get("image_url"), dict) else ""
                if url.startswith("data:image/") and ";base64," in url:
                    try:
                        raw = base64.b64decode(url.split(";base64,", 1)[1], validate=False)
                    except Exception:
                        raw = b""
                    if 0 < len(raw) <= MAX_FRAME_BYTES:
                        img = decode(raw)
        return " ".join(texts).strip()[:300], img

    def make_guide(wcfg):
        from .wonder_agent import WonderGuide

        factory = app.extensions.get("streetsense_guide_factory")
        return factory(wcfg) if factory else WonderGuide(wcfg)

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
