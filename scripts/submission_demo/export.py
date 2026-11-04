"""Mux master and social cut. Writes shareable H264/AAC with full source credits."""
import json,os,re,subprocess
from pathlib import Path
from timeline import SOCIAL
from config import ROOT,FF
R=ROOT/'renders'
credits=('Recorded prototype analysis and original-frame Meta evaluations, not a live navigation trial. '
 'DC footage: Sebastian Wallroth, CC BY 3.0, https://commons.wikimedia.org/wiki/File:Pedestrian_crossing_light_washington_dc.ogv, https://creativecommons.org/licenses/by/3.0/ ; '
 'Tyumen: RG72, CC BY-SA 4.0, https://commons.wikimedia.org/wiki/File:Piedirejo_en_strato_Herzen_(Tjumeno).webm, https://creativecommons.org/licenses/by-sa/4.0/ ; '
 'European signal: Davlat Tursunov, Pexels 6415590, https://www.pexels.com/video/6415590/ , Pexels License; '
 'Tomaszow: WrS.tm.pl, public domain; Frankfurt: ManuelB701, CC0; Mazatlan 3D source: DogeGamer2015MZT, CC0. '
 'Cropped, annotated, retimed and composed with new graphics. Stock Kokoro narration; crossing cues revoiced for the recording. Original synthesized music. Manrope OFL. See CREDITS.md.')
def run(a):subprocess.run([FF,*map(str,a)],check=True)
run(['-y','-v','error','-i',R/'master-picture.mp4','-i',R/'master-audio.wav','-map','0:v','-map','1:a',
 '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-t','180','-movflags','+faststart',
 '-metadata','title=StreetSense — Green is not enough','-metadata','comment='+credits,ROOT/'StreetSense-hackathon-3min.mp4'])
filters=[]
for i,(a,b) in enumerate(SOCIAL):filters.append(f'[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,afade=t=in:d=0.025,afade=t=out:st={b-a-.035}:d=0.035[a{i}]')
filters.append(''.join(f'[a{i}]' for i in range(len(SOCIAL)))+f'concat=n={len(SOCIAL)}:v=0:a=1[out]')
run(['-y','-v','error','-i',R/'master-audio.wav','-filter_complex',';'.join(filters),'-map','[out]',R/'social-audio-raw.wav'])
probe=subprocess.run([FF,'-hide_banner','-i',str(R/'social-audio-raw.wav'),'-af','loudnorm=I=-14:TP=-1.5:LRA=9:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
s=json.loads(re.findall(r'\{[^{}]+\}',probe.stderr)[-1])
norm=('loudnorm=I=-14:TP=-1.5:LRA=9:measured_I='+s['input_i']+':measured_TP='+s['input_tp']+
 ':measured_LRA='+s['input_lra']+':measured_thresh='+s['input_thresh']+':offset='+s['target_offset']+':linear=true')
run(['-y','-v','error','-i',R/'social-audio-raw.wav','-af',norm,'-ar','48000',R/'social-audio.wav'])
run(['-y','-v','error','-i',R/'social-picture.mp4','-i',R/'social-audio.wav','-map','0:v','-map','1:a',
 '-c:v','copy','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-t',sum(b-a for a,b in SOCIAL),'-movflags','+faststart',
 '-metadata','title=StreetSense — Green is not enough','-metadata','comment='+credits,ROOT/'StreetSense-social-50s.mp4'])
print('Both MP4 exports ready',flush=True)
