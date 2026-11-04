"""Editable 180-second judge film + recomposed 50-second portrait cut.

Only public footage and recorded model outputs are shown. No new API calls.
"""
import argparse,functools,json,math,os,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter,ImageEnhance
from timeline import COPY,DURATION,SHOTS,SOCIAL,AUTHOR,DESCRIPTION

from config import ROOT,FF
OUT=ROOT/'renders';A=ROOT/'assets';C=ROOT/'captures'
FPS=30;BG='#0a0e13';WHITE='#f5f6f7';GREEN='#2dd36f';AMBER='#efac27';MUTED='#a0aab7';RED='#ff6862'
META=json.loads((A/'recorded-meta.json').read_text());NARR=json.loads((ROOT/'narration.json').read_text())
DATA={p.stem:json.loads(p.read_text()) for p in C.glob('*.json')}
EXPIRY=json.loads((A/'expiry-rule-replay.json').read_text())
caps={};last={}

@functools.lru_cache(200)
def font(size,weight=650):
 f=ImageFont.truetype(str(A/'fonts/Manrope.ttf'),int(size));f.set_variation_by_axes([weight]);return f

def tx(d,xy,text,size=40,color=WHITE,weight=650):d.text(xy,text,font=font(size,weight),fill=color,anchor='lt')
def fit(d,text,size,width,weight=800):
 while d.textlength(text,font=font(size,weight))>width:size-=1
 return size
def wrap(d,text,size,width,weight=650):
 lines=[];line=''
 for word in text.split():
  test=(line+' '+word).strip()
  if line and d.textlength(test,font=font(size,weight))>width:lines.append(line);line=word
  else:line=test
 if line:lines.append(line)
 return lines
def block(d,x,y,text,size,width,color=WHITE,weight=650,leading=1.25):
 lines=wrap(d,text,size,width,weight)
 for i,line in enumerate(lines):tx(d,(x,y+i*size*leading),line,size,color,weight)
 return len(lines)*size*leading

def make_captions():
 result=[]
 for n in NARR:
  words=n['text'].split();chunks=[];line=[]
  for word in words:
   if line and (len(line)>=7 or len(' '.join(line+[word]))>40):
    tail=[]
    while line and line[-1].lower() in {'a','an','the','for','to','of','and','with','in','on'}:tail.insert(0,line.pop())
    if line:chunks.append(' '.join(line))
    line=tail
   line.append(word)
   if word.endswith(('.', '?', '!')):chunks.append(' '.join(line));line=[]
  if line:chunks.append(' '.join(line))
  weights=[len(c)+12*(c.endswith(('.', '?', '!'))) for c in chunks];total=sum(weights);t=n['start']
  for label,w in zip(chunks,weights):
   end=t+n['duration']*w/total;result.append((t,end,label,n['kind']));t=end
 return sorted(result,key=lambda c:c[0])
CAPTIONS=make_captions()

def video(tag,seconds):
 if tag not in caps:caps[tag]=cv2.VideoCapture(str(C/f'{tag}.mp4'))
 cap=caps[tag];n=len(DATA[tag]['rows']);idx=min(n-1,max(0,int(seconds*15)))
 if last.get(tag,(-1,None))[0]==idx:return last[tag][1].copy(),DATA[tag]['rows'][idx]
 prev=last.get(tag,(-2,None))[0]
 if idx!=prev+1:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
 ok,f=cap.read()
 if not ok:raise RuntimeError(f'Cannot read {tag} frame {idx}')
 im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));last[tag]=(idx,im)
 return im.copy(),DATA[tag]['rows'][idx]

def annotated(im,row):
 d=ImageDraw.Draw(im);occupied=[]
 for det in row['detections']:
  if det['label'] not in ['car','truck','bus','person','traffic light']:continue
  x1,y1,x2,y2=det['box'];state=det['state'];color=GREEN if state=='walk' else RED if state=='red' else AMBER if det['label']=='traffic light' else '#b1e7e7'
  d.rectangle((x1,y1,x2,y2),outline=color,width=3)
  label=(state if state in ['walk','red'] else det['label'])+(f" #{det['id']}" if det['id'] is not None else '')
  width=d.textlength(label,font=font(20,750));lx=max(0,min(x1,im.width-width-6));ly=max(0,y1-26)
  bounds=(lx,ly,lx+width+6,ly+25)
  if any(bounds[0]<b[2]+5 and bounds[2]+5>b[0] and bounds[1]<b[3]+5 and bounds[3]+5>b[1] for b in occupied):continue
  occupied.append(bounds);d.rectangle(bounds,fill='#080e0d');tx(d,(lx+3,ly+2),label,20,color,750)
 return im

def scene(t):
 shot=next(s for s in SHOTS if s[0]<=t<s[1]);begin,end,kind,tag,_,_=shot;u=t-begin
 row=None;sample=None;note='Recorded local analysis';media=None
 if kind in ['red','walk','disagree','countdown']:
  sample={'red':'dc_red','walk':'dc_walk','disagree':'tyumen','countdown':'dc_countdown'}[kind]
  media=Image.open(A/f'{sample}-source.jpg').convert('RGB');note='Recorded Muse evaluation · original frame'
 elif kind=='sam':
  media=Image.open(A/'tyumen-sam.jpg').convert('RGB');note='Recorded SAM 3.1 mask · advisory'
  if u<2:
   raw=Image.open(A/'tyumen-source.jpg').convert('RGB');edge=int(media.width*u/2)
   raw.paste(media.crop((0,0,edge,media.height)),(0,0));media=raw
 elif kind=='freshness':
  media=Image.open(A/'dc_walk-source.jpg').convert('RGB');note='Expiry rule replay · recorded input'
 elif tag:
  start={'hook':3.5,'track':4,'hazard':59/15-2.5,'payoff':3}.get(kind,0)
  media,row=video(tag,start+u);media=annotated(media,row)
 return shot,media,row,sample,note

def value(d,x,y,label,text,color=WHITE,w=560,size=58):
 tx(d,(x,y),label,25,MUTED,600);block(d,x,y+40,text,size,w,color,800)

def status(d,kind,row,sample,u,x,y,w,vertical=False):
 gap=146 if not vertical else 138;size=56 if not vertical else 49
 if sample:
  item=META[sample];dec=item['fusion']['decision'];local=item['local'].upper();muse=item['reply']['safe_to_cross'].upper()
  if vertical:
   value(d,x,y,'LOCAL READER',local,RED if local=='RED' else GREEN if local=='WALK' else AMBER,w/2-18,size)
   value(d,x+w/2+12,y,'MUSE SPARK',muse,GREEN if muse=='YES' else RED,w/2-18,size)
   value(d,x,y+gap,'CROSSING CUE',{'go':'WALK CUE','stop':'DO NOT CROSS','unsure':'WAIT'}[dec],GREEN if dec=='go' else AMBER,w,54)
  else:
   value(d,x,y,'LOCAL READER',local,RED if local=='RED' else GREEN if local=='WALK' else AMBER,w,size)
   value(d,x,y+gap,'MUSE SPARK',muse,GREEN if muse=='YES' else RED,w,size)
   value(d,x,y+gap*2,'CROSSING CUE',{'go':'WALK CUE','stop':'DO NOT CROSS','unsure':'WAIT'}[dec],GREEN if dec=='go' else AMBER,w,58)
   block(d,x,y+gap*3+4,'“'+item['spoken']+'”',29,w,WHITE,600)
  return
 if kind=='hook':
  value(d,x,y,'PEDESTRIAN WALK','NOT CONFIRMED',AMBER,w,47)
  value(d,x,y+gap,'CROSSING CUE','WAIT.',AMBER,w,96)
 elif kind=='freshness':
  age=min(10,u*10/12);expired=age>8
  value(d,x,y,'EVIDENCE AGE',f'{age:.1f} seconds',AMBER if expired else WHITE,w,size)
  value(d,x,y+gap,'8-SECOND LIMIT','EXPIRED' if expired else 'FRESH',AMBER if expired else GREEN,w,size)
  if not vertical:
   speech=EXPIRY[min(40,round(age*4))]['spoken'];block(d,x,y+gap*2,speech,35,w,AMBER if expired else WHITE)
   tx(d,(x,y+gap*3+20),'479 TESTS PASSED · 1 SKIPPED',25,MUTED)
 elif kind=='sam':
  value(d,x,y,'SEGMENT ANYTHING 3.1','GROUNDED MASKS',GREEN,w,45)
  value(d,x,y+gap,'PROMPT','“crosswalk”',WHITE,w,size)
  value(d,x,y+gap*2,'CROSSING AUTHORITY','NONE',AMBER,w,size)
 elif kind=='europe':
  value(d,x,y,'SIGNAL PROFILE','UK / EUROPE',GREEN,w,size)
  value(d,x,y+gap,'CURRENT LOCAL READING',row['local'].upper(),GREEN if row['local']=='walk' else AMBER,w,size)
  value(d,x,y+gap*2,'FOOTAGE','European-style',WHITE,w,42)
 elif kind=='architecture':
  value(d,x,y,'COMPUTER VISION','Detect · track · read',GREEN,w,39)
  value(d,x,y+gap,'META AI','Reason · segment',GREEN,w,40)
  value(d,x,y+gap*2,'SPEECH','Prioritized cues',GREEN,w,43)
 else:
  ds=row['detections'] if row else []
  vehicles=sum(a['label'] in ['car','truck','bus'] for a in ds);people=sum(a['label']=='person' for a in ds)
  value(d,x,y,'TRACKED IN THIS FRAME',f'{vehicles} vehicles',WHITE,w,size)
  value(d,x,y+gap,'PEDESTRIANS',str(people),WHITE,w,size)
  if not vertical:
   spoken=row.get('spoken') if row else None
   if not spoken and row:
    records=DATA['sidewalk' if kind=='problem' else 'tyumen']['rows'];idx=records.index(row)
    spoken=next((r['spoken'] for r in reversed(records[max(0,idx-25):idx+1]) if r['spoken']),None)
   block(d,x,y+gap*2,'“'+spoken+'”' if spoken else 'LOCAL VISION ACTIVE',34,w,GREEN,650)

def caption(d,t,w,h,vertical):
 c=next((c for c in CAPTIONS if c[0]<=t<c[1]),None)
 if not c:return
 size=60 if vertical else 52;lines=wrap(d,c[2],size,w-112,700)
 if len(lines)>2:raise ValueError(('Caption overflow',c))
 yy=1625 if vertical else 942
 for i,line in enumerate(lines):
  width=d.textlength(line,font=font(size,700));tx(d,((w-width)/2,yy+i*size*1.28),line,size,AMBER if c[3]=='cue' else WHITE,700)
 if c[3]=='cue':
  label='Cue text revoiced for recording';size2=24 if vertical else 22
  tw=d.textlength(label,font=font(size2,600));tx(d,((w-tw)/2,yy-38),label,size2,MUTED,600)

@functools.lru_cache(2)
def title_background(w,h):
 src=Image.open(A/'tyumen-source.jpg').convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
 src=src.filter(ImageFilter.GaussianBlur(26))
 return ImageEnhance.Brightness(src).enhance(.20)

def render_frame(t,vertical=False):
 w,h=(1080,1920) if vertical else (1920,1080)
 shot,media,row,sample,note=scene(t);a,b,kind,tag,title,accent=shot;u=t-a
 im=Image.new('RGB',(w,h),BG);d=ImageDraw.Draw(im)
 if kind in ['intro','end']:
  im=title_background(w,h).copy();d=ImageDraw.Draw(im)
  x=60 if vertical else 104
  yy=(565 if vertical else 276)+round(18*max(0,1-u/.5))
  fs=fit(d,'StreetSense',146 if vertical else 156,w-2*x)
  tx(d,(x,yy),'StreetSense',fs,WHITE,800)
  if kind=='intro':
   block(d,x,yy+(205 if vertical else 210),DESCRIPTION,52 if vertical else 47,w-2*x,WHITE,550,1.3)
   tx(d,(x,yy+(455 if vertical else 330)),f'Created by {AUTHOR}',39 if vertical else 37,GREEN,650)
   block(d,x,yy+(550 if vertical else 405),'Meta AI Developer Hackathon 2026 · Prototype',26,w-2*x,MUTED,550)
   caption(d,t,w,h,vertical)
  else:
   block(d,x,yy+(235 if vertical else 205),'Hear what matters.',70 if vertical else 68,w-2*x,GREEN,700)
   tx(d,(x,yy+(415 if vertical else 330)),f'Created by {AUTHOR}',36,WHITE,600)
   block(d,x,yy+(540 if vertical else 430),'github.com/Alexeiguch/streetsense',31 if vertical else 40,w-2*x,MUTED,550)
 if kind=='end':
  credits=['DC: Sebastian Wallroth / CC BY 3.0 · Tyumen: RG72 / CC BY-SA 4.0',
           'European signal: Davlat Tursunov / Pexels · Tomaszów: WrS.tm.pl / Public domain',
           'Frankfurt: ManuelB701 / CC0']
  for i,line in enumerate(credits):block(d,60 if vertical else 104,(1740 if vertical else 968)+i*(43 if vertical else 29),line,19 if vertical else 22,w-130,MUTED,500)
 elif kind!='intro':
  tx(d,(54 if vertical else 64,65 if vertical else 34),'StreetSense',36,WHITE,750)
  tx(d,(760 if vertical else 1550,76 if vertical else 42),AUTHOR if kind=='hook' else 'PROTOTYPE',24,MUTED)
  if vertical:
   yy=175+round(15*max(0,1-u/.4));sz=fit(d,title,104,972)
   tx(d,(54,yy),title,sz,WHITE,850)
   if accent:
    for i,line in enumerate(wrap(d,accent,85,972,750)):tx(d,(54,yy+142+i*108),line,85,GREEN if kind not in ['hook','disagree','countdown','freshness'] else AMBER,750)
   x,y,vw,vh=54,550,972,547
  else:
   text=title if kind=='hook' and u<2.2 else (title+' '+accent).strip()
   color=GREEN if kind=='hook' and u<2.2 else AMBER if kind in ['hook','disagree','countdown','freshness'] else WHITE
   sz=fit(d,text,74,1792);tx(d,(64,110+round(12*max(0,1-u/.4))),text,sz,color,750)
   x,y,vw,vh=64,242,1152,648
  if media:
   media=media.resize((vw,vh),Image.Resampling.LANCZOS);im.paste(media,(x,y));d=ImageDraw.Draw(im)
   d.rectangle((x-1,y-1,x+vw,y+vh),outline='#26313e',width=1)
  # A static inset is a crop from the actual source frame, not a generated signal.
  if sample in ['dc_red','dc_walk','dc_countdown']:
   src=Image.open(A/f'{sample}-source.jpg').convert('RGB');sw,sh=src.size
   crop=src.crop((int(sw*.27),int(sh*.40),int(sw*.39),int(sh*.75)))
   crop.thumbnail((195,240));ix=x+vw-crop.width-18;iy=y+18
   im.paste(crop,(ix,iy));d=ImageDraw.Draw(im);d.rectangle((ix-2,iy-2,ix+crop.width+2,iy+crop.height+2),outline=GREEN,width=3)
  if vertical:
   block(d,54,1122,note,22,972,MUTED,600)
   status(d,kind,row,sample,u,54,1210,972,True)
  else:
   tx(d,(64,904),note,23,MUTED,600)
   d.line((1240,242,1240,890),fill='#26313e',width=1)
   status(d,kind,row,sample,u,1270,260,586)
  caption(d,t,w,h,vertical)
 if not vertical:d.rectangle((0,h-4,round(w*t/DURATION),h),fill=GREEN)
 # A brief motion slide keeps cuts readable.
 if u<.18:im=Image.blend(Image.new('RGB',(w,h),BG),im,min(1,u/.18))
 return im

def srt(path,segments):
 rows=[];offset=0
 def stamp(t):
  z=round(t*1000);return f'{z//3600000:02}:{z//60000%60:02}:{z//1000%60:02},{z%1000:03}'
 for a,b in segments:
  for start,end,text,kind in CAPTIONS:
   lo=max(a,start);hi=min(b,end)
   if hi>lo:rows.append(f'{len(rows)+1}\n{stamp(offset+lo-a)} --> {stamp(offset+hi-a)}\n{text}')
  offset+=b-a
 path.write_text('\n\n'.join(rows)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--portrait',action='store_true');ap.add_argument('--stills',action='store_true');args=ap.parse_args()
 segs=SOCIAL if args.portrait else [(0,180)];duration=sum(b-a for a,b in segs)
 name='social' if args.portrait else 'master';w,h=(1080,1920) if args.portrait else (1920,1080)
 srt(ROOT/f'{name}-captions.srt',segs)
 if args.stills:
  for t in [3,11,22,34,45,57,72,88,103,115,128,145,154,166,176]:render_frame(t,args.portrait).save(OUT/f'{name}-{t:03}.jpg',quality=92)
  return
 enc=subprocess.Popen([FF,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(FPS),'-i','-',
  '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{name}-picture.mp4')],stdin=subprocess.PIPE)
 count=0
 try:
  for a,b in segs:
   for k in range(round((b-a)*FPS)):
    enc.stdin.write(render_frame(a+k/FPS,args.portrait).tobytes());count+=1
    if count%300==0:print(name,count,'/',round(duration*FPS),flush=True)
 finally:enc.stdin.close()
 if enc.wait():raise RuntimeError('Encoding failed')
 print(name,'picture ready',flush=True)

if __name__=='__main__':main()
