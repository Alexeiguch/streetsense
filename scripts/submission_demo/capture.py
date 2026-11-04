"""Capture real local pipeline outputs. No Meta calls; recorded Meta cases are separate."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import cv2

from config import PROJECT_ROOT as ROOT,ROOT as DEMO_ROOT,FF
OUT=DEMO_ROOT/'captures'
sys.path.insert(0,str(ROOT))
from streetsense.config import Config
from streetsense.pipeline import Pipeline
from streetsense.signal_trace import summarise_signals

FPS=15
CASES={
 'vehicle':('samples/heldout/ffm_vehicle.mp4',1,7,'auto'),
 'dc_red':('samples/dc_ped_light.ogv',1,5,'auto'),
 'dc_walk':('samples/dc_ped_light.ogv',10,17,'auto'),
 'countdown':('samples/dc_ped_light.ogv',48,54,'auto'),
 'tyumen':('samples/tyumen_crossing.webm',0,17,'auto'),
 'europe':('samples/heldout/pexels_6415590.mp4',0,10,'GB'),
 'sidewalk':('samples/tomaszow_scooter.ogv',0,11,'auto'),
}

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for name,(filename,start,end,country) in CASES.items():
  if (OUT/f'{name}.json').exists():
   print('Using existing capture',name,flush=True);continue
  cfg=Config();cfg.vlm.enabled=False;cfg.sam.enabled=False;cfg.reader.country=country
  clock=type('Clock',(),{'t':0.,'__call__':lambda self:self.t})()
  pipe=Pipeline(cfg,use_surface=name in {'tyumen','sidewalk'},clock=clock)
  cap=cv2.VideoCapture(str(ROOT/filename));fps=cap.get(cv2.CAP_PROP_FPS)
  enc=subprocess.Popen([FF,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x720','-r',str(FPS),'-i','-',
      '-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/f'{name}.mp4')],stdin=subprocess.PIPE)
  rows=[]
  n=round((end-start)*FPS)
  try:
   for j in range(-FPS if start>=1 else 0,n):
    requested=start+j/FPS;idx=round(requested*fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,f=cap.read()
    if not ok:break
    f=cv2.resize(f,(1280,720),interpolation=cv2.INTER_AREA)
    stamp=idx/fps;clock.t=stamp;result=pipe.process(f,annotate=False,t=stamp)
    if j<0:continue
    trace=summarise_signals(result,1280,720)
    rows.append({'source_frame':idx,'source_s':stamp,'local':trace['local'],'signal_trace':trace,
      'detections':[{'label':d.label,'id':d.track_id,'box':[d.box.x1,d.box.y1,d.box.x2,d.box.y2],
                     'state':d.attrs.get('light_state'),'confidence':float(d.conf)} for d in result.detections],
      'spoken':result.spoken.text if result.spoken else None,
      'alerts':[{'text':a.text,'priority':int(a.priority)} for a in result.candidates]})
    enc.stdin.write(f.tobytes())
   cap.release()
  finally:
   enc.stdin.close()
  if enc.wait():raise RuntimeError(name+' encoding failed')
  blob={'source':filename,'source_sha256':hashlib.sha256((ROOT/filename).read_bytes()).hexdigest(),
        'fps':FPS,'country':country,'meta_calls':0,'recorded_analysis':True,'rows':rows}
  (OUT/f'{name}.json').write_text(json.dumps(blob,separators=(',',':')))
  print(name,len(rows),sorted(set(x['local'] for x in rows)),
        sorted(set(x['spoken'] for x in rows if x['spoken'])),flush=True)

if __name__=='__main__':main()
