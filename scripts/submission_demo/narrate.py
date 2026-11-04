"""Local stock neural voices. No cloning, no remote speech service."""
import json,os,wave
from pathlib import Path
import numpy as np
from kokoro_onnx import Kokoro
from timeline import COPY

from config import ROOT
OUT=ROOT/'renders';OUT.mkdir(exist_ok=True)
MODEL=Path(os.environ.get('KOKORO_MODELS','/private/tmp/streetsense-neural-models'))
k=Kokoro(str(MODEL/'kokoro-v1.0.onnx'),str(MODEL/'voices-v1.0.bin'))
stats=[]
for i,(start,end,text,kind) in enumerate(COPY):
 voice='af_heart' if kind=='narrator' else 'am_michael'
 path=OUT/f'voice-{i:02}.wav'
 if path.exists():
  with wave.open(str(path),'rb') as f:duration=f.getnframes()/f.getframerate()
 else:
  speed=1.0
  for attempt in range(3):
   a,sr=k.create(text,voice=voice,speed=speed,lang='en-us')
   a=np.asarray(a);nz=np.flatnonzero(abs(a)>.002)
   a=a[max(0,nz[0]-round(.035*sr)):min(len(a),nz[-1]+round(.10*sr))]
   duration=len(a)/sr
   if duration<=end-start-.08:break
   speed*=duration/(end-start-.10)
  if duration>end-start-.02:raise ValueError(f'Line {i} exceeds its slot; shorten the script')
  with wave.open(str(path),'wb') as f:
   f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr)
   f.writeframes((np.clip(a,-.999,.999)*32767).astype('<i2').tobytes())
 stats.append({'line':i,'start':start,'end':start+duration,'text':text,'kind':kind,'voice':voice,'duration':duration})
 print(i,round(duration,2),text,flush=True)
(ROOT/'narration.json').write_text(json.dumps(stats,indent=2))
