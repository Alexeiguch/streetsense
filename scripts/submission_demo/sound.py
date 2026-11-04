"""Original synthesized score + local neural speech, two-pass loudness master."""
import json,math,os,re,subprocess,wave
from pathlib import Path
import numpy as np
from timeline import DURATION,SHOTS,SOCIAL
from config import ROOT,FF
OUT=ROOT/'renders';SR=48000;N=SR*DURATION

def write(path,data):
 with wave.open(str(path),'wb') as f:
  f.setnchannels(2);f.setsampwidth(2);f.setframerate(SR)
  f.writeframes((np.clip(data,-.999,.999)*32767).astype('<i2').tobytes())

music=np.zeros((N,2),dtype=np.float32);voice=np.zeros_like(music);duck=np.ones(N,dtype=np.float32)
rng=np.random.default_rng(261104)
def add(t,x,pan=0.):
 start=round(t*SR);n=min(len(x),N-start)
 if n<=0:return
 music[start:start+n,0]+=x[:n]*math.sqrt((1-pan)/2)
 music[start:start+n,1]+=x[:n]*math.sqrt((1+pan)/2)
def note(t,f,length,level=.04,pan=0.):
 z=np.arange(round(length*SR),dtype=np.float32)/SR
 env=(1-np.exp(-z*45))*np.exp(-z*2.4)*np.minimum(1,(length-z)*8)
 sig=(np.sin(2*np.pi*f*z)+.22*np.sin(4*np.pi*f*z))*env*level;add(t,sig,pan)
chords=[(146.83,174.61,220,293.66),(116.54,174.61,233.08,293.66),
        (130.81,164.81,196,261.63),(110,164.81,220,277.18)]
for i,t in enumerate(np.arange(0,DURATION-.2,.5)):
 chord=chords[(i//16)%4];level=.035 if t>=7 else .015
 note(t,chord[(i*3)%4]*2,1.5,level,(-.4,.4)[i%2])
 if i%2==0:
  note(t,chord[0]/2,.55,.05)
  z=np.arange(int(.20*SR))/SR
  add(t,np.sin(2*np.pi*(47*z+35*.04*(1-np.exp(-z/.04))))*np.exp(-z*24)*(.055 if t>=7 else .025))
 if t>=7:
  z=np.arange(int(.025*SR))/SR
  tick=rng.normal(0,1,len(z));tick=np.r_[0,np.diff(tick)]*np.exp(-z*180)*.006
  add(t+.25,tick,(-.45,.45)[i%2])
for cut,*_ in SHOTS:
 note(cut,587.33,.65,.04,-.15);note(cut+.09,880,.5,.025,.15)
for i,n in enumerate(json.loads((ROOT/'narration.json').read_text())):
 with wave.open(str(OUT/f'voice-{i:02}.wav'),'rb') as f:
  rate=f.getframerate();x=np.frombuffer(f.readframes(f.getnframes()),'<i2').astype(np.float32)/32768
 x=np.interp(np.arange(round(len(x)*SR/rate))/SR,np.arange(len(x))/rate,x).astype(np.float32)
 x*=.58/max(.01,float(abs(x).max()));a=round(n['start']*SR);b=min(N,a+len(x));voice[a:b]+=x[:b-a,None]
 lo=max(0,a-round(.08*SR));hi=min(N,b+round(.16*SR));duck[lo:hi]=np.minimum(duck[lo:hi],.12 if n['kind']=='cue' else .24)
# Smooth sidechain boundaries with a short low-cost moving average.
kernel=round(.015*SR);cs=np.cumsum(np.pad(duck,(kernel//2,kernel-kernel//2),mode='edge'),dtype=np.float64)
duck=((cs[kernel:]-cs[:-kernel])/kernel).astype(np.float32)
t=np.arange(N,dtype=np.float32)/SR;fade=np.minimum(1,t/.15)*np.minimum(1,(DURATION-t)/.65)
mixed=(music*duck[:,None]+voice)*fade[:,None]
write(OUT/'score.wav',music*fade[:,None]);write(OUT/'narration.wav',voice);write(OUT/'mix.wav',mixed)
first=subprocess.run([FF,'-hide_banner','-i',str(OUT/'mix.wav'),'-af','loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
s=json.loads(re.findall(r'\{[^{}]+\}',first.stderr)[-1])
filt=('loudnorm=I=-14:TP=-1.5:LRA=9:measured_I='+s['input_i']+':measured_TP='+s['input_tp']+
 ':measured_LRA='+s['input_lra']+':measured_thresh='+s['input_thresh']+':offset='+s['target_offset']+':linear=true:print_format=json')
second=subprocess.run([FF,'-y','-hide_banner','-i',str(OUT/'mix.wav'),'-af',filt,'-ar',str(SR),'-c:a','pcm_s24le',str(OUT/'master-audio.wav')],capture_output=True,text=True,check=True)
result=json.loads(re.findall(r'\{[^{}]+\}',second.stderr)[-1])
(ROOT/'audio-checks.json').write_text(json.dumps({'normalization':result,'listening_review':'Not available in this host; objective checks only.','narration':'Stock Kokoro voices; cue text revoiced from recorded renderer outputs.'},indent=2))
print(json.dumps(result),flush=True)
