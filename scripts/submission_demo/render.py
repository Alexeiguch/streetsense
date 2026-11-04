"""Editable 180-second judge film + recomposed 50-second portrait cut.

Only public footage and recorded model outputs are shown. No new API calls.
"""
import argparse,functools,json,math,os,subprocess
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from timeline import COPY,DURATION,SHOTS,SOCIAL

from config import ROOT,FF
OUT=ROOT/'renders';A=ROOT/'assets';C=ROOT/'captures'
FPS=30;BG='#080e0d';WHITE='#f4f7f3';GREEN='#9ef57c';AMBER='#ffc475';MUTED='#a8b8b0';RED='#ff8d7d'
META=json.loads((A/'recorded-meta.json').read_text());NARR=json.loads((ROOT/'narration.json').read_text())
DATA={p.stem:json.loads(p.read_text()) for p in C.glob('*.json')}
EXPIRY=json.loads((A/'expiry-rule-replay.json').read_text())
cloud=np.load(A/'cloud-demo.npz');caps={};last={}

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
   if line and (len(line)>=7 or len(' '.join(line+[word]))>43):chunks.append(' '.join(line));line=[]
   line.append(word)
   if word.endswith(('.', '?', '!')):chunks.append(' '.join(line));line=[]
  if line:chunks.append(' '.join(line))
  weights=[len(c)+12*(c.endswith(('.', '?', '!'))) for c in chunks];total=sum(weights);t=n['start']
  for label,w in zip(chunks,weights):
   end=t+n['duration']*w/total;result.append((t,end,label,n['kind']));t=end
 return result
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
 d=ImageDraw.Draw(im)
 for det in row['detections']:
  if det['label'] not in ['car','truck','bus','person','traffic light']:continue
  x1,y1,x2,y2=det['box'];state=det['state'];color=GREEN if state=='walk' else RED if state=='red' else AMBER if det['label']=='traffic light' else '#b1e7e7'
  d.rectangle((x1,y1,x2,y2),outline=color,width=3)
  label=(state or det['label'])+(f" #{det['id']}" if det['id'] is not None else '')
  tx(d,(x1,max(0,y1-24)),label,20,color,750)
 return im

def cloud_image(t,w,h):
 p=cloud['points'];a=math.radians(-26+58*math.sin(t*.16));b=math.radians(-12)
 x=p[:,0]*math.cos(a)+p[:,2]*math.sin(a);z=-p[:,0]*math.sin(a)+p[:,2]*math.cos(a)
 y=p[:,1]*math.cos(b)-z*math.sin(b);order=np.argsort(z)[::-1]
 xx=(w/2+x*min(w,h)*.68).astype(int);yy=(h/2+y*min(w,h)*.68).astype(int)
 good=(xx>=1)&(xx<w-2)&(yy>=1)&(yy<h-2);ids=order[good[order]]
 rgb=np.zeros((h,w,3),np.uint8);rgb[:]=(16,24,23)
 for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:rgb[yy[ids]+dy,xx[ids]+dx]=cloud['colors'][ids]
 return Image.fromarray(rgb)

def scene(t):
 shot=next(s for s in SHOTS if s[0]<=t<s[1]);begin,end,kind,tag,_,_=shot;u=t-begin
 row=None;sample=None;note='RECORDED LOCAL ANALYSIS';media=None
 if kind in ['red','walk','disagree','countdown']:
  sample={'red':'dc_red','walk':'dc_walk','disagree':'tyumen','countdown':'dc_countdown'}[kind]
  media=Image.open(A/f'{sample}-source.jpg').convert('RGB');note='RECORDED MUSE EVALUATION · ORIGINAL INPUT FRAME'
 elif kind=='sam':
  media=Image.open(A/'tyumen-sam.jpg').convert('RGB');note='RECORDED SAM 3.1 EVALUATION · ADVISORY MASK'
  if u<2:
   raw=Image.open(A/'tyumen-source.jpg').convert('RGB');edge=int(media.width*u/2)
   raw.paste(media.crop((0,0,edge,media.height)),(0,0));media=raw
 elif kind=='freshness':
  media=Image.open(A/'dc_walk-source.jpg').convert('RGB');note='EXPIRY TEST REPLAY · NOT A LIVE CROSSING'
 elif kind=='3d':note='ACTUAL MAPANYTHING OUTPUT · OFFLINE EXPERIMENT'
 elif tag:
  start={'hook':3.5,'track':4,'payoff':3}.get(kind,0)
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
 elif kind=='3d':
  value(d,x,y,'META MAPANYTHING','3 VIEWS',GREEN,w,size)
  value(d,x,y+gap,'MEASURED INFERENCE','3.2 s / batch',WHITE,w,48)
  value(d,x,y+gap*2,'DECISION PATH','OFFLINE ONLY',AMBER,w,47)
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
    records=DATA['sidewalk' if kind=='intro' else 'tyumen']['rows'];idx=records.index(row)
    spoken=next((r['spoken'] for r in reversed(records[max(0,idx-25):idx+1]) if r['spoken']),None)
   block(d,x,y+gap*2,'“'+spoken+'”' if spoken else 'LOCAL VISION ACTIVE',34,w,GREEN,650)

def caption(d,t,w,h,vertical):
 c=next((c for c in CAPTIONS if c[0]<=t<c[1]),None)
 if not c:return
 size=57 if vertical else 49;lines=wrap(d,c[2],size,w-112,650)
 if len(lines)>2:raise ValueError(('Caption overflow',c))
 yy=1625 if vertical else 942
 for i,line in enumerate(lines):
  width=d.textlength(line,font=font(size,650));tx(d,((w-width)/2,yy+i*size*1.28),line,size,AMBER if c[3]=='cue' else WHITE,650)
 if c[3]=='cue':
  label='SPOKEN CUE · REVOICED FOR RECORDING';size2=24 if vertical else 22
  tw=d.textlength(label,font=font(size2,600));tx(d,((w-tw)/2,yy-38),label,size2,MUTED,600)

def render_frame(t,vertical=False):
 w,h=(1080,1920) if vertical else (1920,1080)
 shot,media,row,sample,note=scene(t);a,b,kind,tag,title,accent=shot;u=t-a
 im=Image.new('RGB',(w,h),BG);d=ImageDraw.Draw(im)
 if kind=='end':
  for j in range(7):
   xx=(55+j*145) if vertical else (120+j*240);yy=(300 if vertical else 90)+math.sin(u*.7+j*.35)*20
   bw=62 if vertical else 102;hh=210 if vertical else 170
   d.polygon([(xx,yy),(xx+bw,yy-20),(xx+bw+22,yy+hh),(xx+22,yy+hh+20)],fill=GREEN if j==3 else WHITE)
  fs=fit(d,'StreetSense.',146 if vertical else 205,w-110)
  tx(d,(54 if vertical else 95,725 if vertical else 365),'StreetSense.',fs,WHITE,850)
  block(d,60 if vertical else 108,970 if vertical else 610,'HEAR WHAT MATTERS.',70 if vertical else 74,w-120,GREEN,800)
  tx(d,(60 if vertical else 108,1210 if vertical else 760),'Explore the prototype',43,WHITE)
  block(d,60 if vertical else 108,1300 if vertical else 829,'github.com/Alexeiguch/streetsense',31 if vertical else 38,w-130,MUTED,600)
  credits=['DC: Sebastian Wallroth / CC BY 3.0 · Tyumen: RG72 / CC BY-SA 4.0',
           'European signal: Davlat Tursunov / Pexels · Tomaszów: WrS.tm.pl / Public domain',
           'Frankfurt: ManuelB701 / CC0 · 3D source: DogeGamer2015MZT / CC0']
  for i,line in enumerate(credits):block(d,60 if vertical else 108,(1740 if vertical else 968)+i*(43 if vertical else 29),line,19 if vertical else 22,w-130,MUTED,500)
 else:
  tx(d,(54 if vertical else 64,65 if vertical else 34),'StreetSense',36,GREEN,800)
  tx(d,(805 if vertical else 1550,76 if vertical else 42),'PROTOTYPE',24,MUTED)
  if vertical:
   yy=175+round(15*max(0,1-u/.4));sz=fit(d,title,110,972)
   tx(d,(54,yy),title,sz,WHITE,850)
   if accent:
    for i,line in enumerate(wrap(d,accent,90,972,850)):tx(d,(54,yy+142+i*112),line,90,GREEN if kind not in ['hook','disagree','countdown','freshness'] else AMBER,850)
   x,y,vw,vh=54,550,972,547
  else:
   text=title+' '+accent;sz=fit(d,text,81,1792);tx(d,(64,110+round(12*max(0,1-u/.4))),text,sz,AMBER if kind in ['hook','disagree','countdown','freshness'] else WHITE,850)
   x,y,vw,vh=64,242,1152,648
  if kind=='3d':media=cloud_image(u,vw,vh)
  if media:
   media=media.resize((vw,vh),Image.Resampling.LANCZOS);im.paste(media,(x,y));d=ImageDraw.Draw(im)
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
   status(d,kind,row,sample,u,1270,260,586)
  caption(d,t,w,h,vertical)
 # A brief motion slide, not flashes, keeps cuts readable.
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
