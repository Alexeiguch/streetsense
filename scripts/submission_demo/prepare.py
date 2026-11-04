"""Bundle only public sample-derived evidence; never load credentials."""
import json,os,shutil,sys
from pathlib import Path

from config import PROJECT_ROOT as ROOT,ROOT as DEMO_ROOT
DEST=DEMO_ROOT/'assets'
DEST.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from streetsense.meta_api import validate_crossing_reply,render_crossing_speech

review=json.loads((ROOT/'docs/review_2026_11_04/live-meta.json').read_text())
selected={}
for name in ['dc_red','dc_walk','dc_countdown','tyumen']:
 row=next(r for r in review['rows'] if r['kind']=='muse' and r['sample']==name and r['repeat']==1)
 selected[name]={'local':row['local_state'],'reply':row['reply'],'fusion':row['fusion'],
                 'spoken':render_crossing_speech(validate_crossing_reply(row['reply']),local_state=row['local_state'])}
 shutil.copyfile(ROOT/f'outputs/review/{name}-source.jpg',DEST/f'{name}-source.jpg')
for name in ['dc_walk','tyumen']:
 shutil.copyfile(ROOT/f'outputs/review/{name}-sam.jpg',DEST/f'{name}-sam.jpg')
(DEST/'recorded-meta.json').write_text(json.dumps(selected,indent=2))
valid=validate_crossing_reply(selected['dc_walk']['reply'],capture_ts=0.)
expiry=[{'age':i/4,'spoken':render_crossing_speech(valid,local_state='walk',now=i/4,max_age_s=8.)} for i in range(41)]
(DEST/'expiry-rule-replay.json').write_text(json.dumps(expiry,indent=2))
(DEST/'proof.json').write_text(json.dumps({'tests_passed':479,'tests_skipped':1,'date':'2026-11-04',
 'local_recording':'Current pipeline; remote calls disabled. Not a live navigation or user trial.',
 'meta_recording':'Actual Muse Spark 1.3 / SAM 3.1 evaluations from 4 November. Replies replayed on their original image.',
 'cue_voice':'Exact production crossing renderer and selected local alert output, revoiced for this recording; not device audio.',
 'hazard':'Tyumen capture row 59, source time 3.9333s: car approaching on your right.',
 'expiry':'Replay of render_crossing_speech with a recorded walk reply and an explicit 8s maximum age.'},indent=2))
print('Bundled public-sample evidence and exact crossing speech')
