"""Check actual encoded exports, extract proof frames and measure decoded AAC audio."""
import hashlib,json,os,re,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
from config import ROOT,FF
R=ROOT/'renders'
items=[('StreetSense-hackathon-3min.mp4',180,(1920,1080),5400),('StreetSense-social-50s.mp4',50,(1080,1920),1500)]
checks=[]
for name,duration,size,count in items:
 path=ROOT/name
 p=subprocess.run([FF,'-hide_banner','-i',str(path),'-map','0:v:0','-map','0:a:0','-f','null','-'],capture_output=True,text=True)
 assert p.returncode==0,(name,p.stderr[-1000:])
 assert f'{size[0]}x{size[1]}' in p.stderr
 assert 'Video: h264' in p.stderr and 'yuv420p' in p.stderr
 assert 'Audio: aac' in p.stderr and '48000 Hz' in p.stderr
 counts=re.findall(r'frame=\s*(\d+)',p.stderr);assert int(counts[-1])==count,(name,counts[-1])
 match=re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)',p.stderr);hh,mm,ss=map(float,match.groups());actual=hh*3600+mm*60+ss
 assert abs(actual-duration)<=.02,(actual,duration)
 raw=path.read_bytes();assert raw.index(b'moov')<raw.index(b'mdat'),'Faststart missing'
 q=subprocess.run([FF,'-hide_banner','-i',str(path),'-af','loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
 audio=json.loads(re.findall(r'\{[^{}]+\}',q.stderr)[-1]);assert float(audio['input_tp'])<=-1.0
 assert -16<=float(audio['input_i'])<=-13.8,(name,audio['input_i'])
 checks.append({'file':name,'duration_s':actual,'size':size,'frames':count,'fps':30,'video':'h264/yuv420p','audio':'aac/48000/stereo',
   'full_decode':'pass','faststart':True,'bytes':path.stat().st_size,'sha256':hashlib.sha256(raw).hexdigest(),
   'measured_lufs':audio['input_i'],'measured_true_peak_dbtp':audio['input_tp']})
 print(json.dumps(checks[-1]),flush=True)
times=[2.4,11,22,34,45,57,72,88,103,115,128,147.6,154,166,176]
frames=[]
for t in times:
 path=R/f'encoded-{t:.1f}.jpg'
 subprocess.run([FF,'-y','-v','error','-ss',str(t),'-i',str(ROOT/items[0][0]),'-frames:v','1','-q:v','2',str(path)],check=True)
 im=Image.open(path);im.thumbnail((480,270));frames.append(im.copy())
sheet=Image.new('RGB',(1440,1350))
for i,im in enumerate(frames):sheet.paste(im,((i%3)*480,(i//3)*270))
sheet.save(ROOT/'review-contact-sheet.jpg',quality=94)
subprocess.run([FF,'-y','-v','error','-ss','2.4','-i',str(ROOT/items[0][0]),'-frames:v','1','-q:v','2',str(ROOT/'thumbnail.jpg')],check=True)
allowed={x['spoken'] for x in json.loads((ROOT/'assets/recorded-meta.json').read_text()).values()}
for n in json.loads((ROOT/'narration.json').read_text()):
 assert n['end']<=180
 if n['kind']=='cue':assert n['text'] in allowed
result={'exports':checks,'cue_text_matches_recorded_renderer':True,'encoded_frames_extracted':len(times),
 'caption_timing':'Phrase timing estimated from generated take durations; not word alignment.',
 'subjective_listening_review':'Unavailable in this host; requires viewer review.',
 'continuous_motion_review':'Full decode and browser playback checked separately; still inspection alone is not full-motion quality review.'}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
