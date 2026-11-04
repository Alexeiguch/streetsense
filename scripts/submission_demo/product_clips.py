"""Export silent editing assets from saved product outputs, without a promo layout."""
import json, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

from config import PROJECT_ROOT, ROOT, FF

sys.path.insert(0, str(PROJECT_ROOT))
from streetsense.overlay import draw
from streetsense.types import Alert, Box, Detection, Priority

OUT = PROJECT_ROOT / 'outputs/product-videos'
W, H, FPS = 1280, 720, 15
CLIPS = [
    ('tyumen', '01-detection-tracking-hazard-cues'),
    ('sidewalk', '02-sidewalk-obstacle-tracking'),
    ('europe', '03-uk-europe-signal-reader'),
    ('vehicle', '04-vehicle-green-is-not-pedestrian-walk'),
    ('dc_red', '05-red-hand-local-reader'),
    ('dc_walk', '06-walk-local-reader'),
    ('countdown', '07-countdown-local-reader'),
]


def encoder(name):
    return subprocess.Popen([
        FF, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / name),
    ], stdin=subprocess.PIPE)


def stamp(frame, text):
    cv2.rectangle(frame, (0, H - 32), (W, H), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (245, 245, 245), 1, cv2.LINE_AA)
    return frame


def finish(enc):
    enc.stdin.close()
    if enc.wait():
        raise RuntimeError('Encoding failed')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tag, name in CLIPS:
        data = json.loads((ROOT / f'captures/{tag}.json').read_text())
        cap = cv2.VideoCapture(str(ROOT / f'captures/{tag}.mp4'))
        enc = encoder(name + '.mp4')
        recent, events = [], []
        for i, row in enumerate(data['rows']):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'{tag}: missing frame {i}')
            t = i / FPS
            heads = {h['track_id']: h for h in row['signal_trace']['heads']}
            detections = [Detection(
                d['label'], Box(*d['box']), d['confidence'], d['id'],
                {'light_state': d['state'] or 'unknown',
                 'signal_kind': heads.get(d['id'], {}).get('kind', '?')},
            ) for d in row['detections']]
            if row.get('spoken'):
                priority = next((a['priority'] for a in row['alerts']
                                 if a['text'] == row['spoken']), 1)
                recent.insert(0, Alert(str(i), row['spoken'], Priority(priority), t))
                events.append({'clip_s': t, 'source_s': row['source_s'],
                               'text': row['spoken']})
            recent = [a for a in recent if t - a.ts <= 2.5]
            frame = draw(frame, detections, None, None, recent)
            stamp(frame, f"Recorded local output | local: {row['local']} | "
                  f"profile: {data['country']} | Meta not called")
            enc.stdin.write(np.ascontiguousarray(frame).tobytes())
        cap.release()
        finish(enc)
        manifest.append({'file': name + '.mp4', 'frames': len(data['rows']),
                         'duration_s': len(data['rows']) / FPS,
                         'kind': 'recorded local pipeline output',
                         'source': data['source'], 'selected_cue_text': events})
        print(name, flush=True)

    meta = json.loads((ROOT / 'assets/recorded-meta.json').read_text())
    cases = [('dc_red', '08-meta-red-hand-stop'),
             ('dc_walk', '09-meta-walk-two-source-agreement'),
             ('tyumen', '10-meta-disagreement-blocks-go'),
             ('dc_countdown', '11-meta-one-second-countdown-stop')]
    for tag, name in cases:
        item = meta[tag]
        image = cv2.imread(str(ROOT / f'assets/{tag}-source.jpg'))
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        alert = Alert('recorded', item['spoken'],
                      Priority.NOTICE if item['fusion']['decision'] == 'go'
                      else Priority.CRITICAL)
        frame = draw(image, [], None, None, [alert],
                     vlm_text=item['reply']['summary'])
        stamp(frame, f"Recorded evaluation - original input frame | local: {item['local']} | "
              f"Muse: {item['reply']['safe_to_cross']} | cue: {item['fusion']['decision']}")
        enc = encoder(name + '.mp4')
        for _ in range(5 * FPS):
            enc.stdin.write(frame.tobytes())
        finish(enc)
        manifest.append({'file': name + '.mp4', 'frames': 5 * FPS,
                         'duration_s': 5, 'kind': 'static recorded Meta evaluation',
                         'evidence': item})
        print(name, flush=True)

    raw = cv2.resize(cv2.imread(str(ROOT / 'assets/tyumen-source.jpg')), (W, H))
    mask = cv2.resize(cv2.imread(str(ROOT / 'assets/tyumen-sam.jpg')), (W, H))
    name = '12-sam-crosswalk-segmentation.mp4'
    enc = encoder(name)
    for i in range(6 * FPS):
        frame = raw.copy()
        edge = round(W * min(1, i / (2 * FPS)))
        frame[:, :edge] = mask[:, :edge]
        stamp(frame, 'Recorded SAM 3.1 output | prompt: crosswalk | advisory mask - no crossing authority')
        enc.stdin.write(frame.tobytes())
    finish(enc)
    manifest.append({'file': name, 'frames': 6 * FPS, 'duration_s': 6,
                     'kind': 'recorded mask; editorial reveal over original frame'})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    credits = (ROOT / 'CREDITS.md').read_text().split('## Footage\n\n', 1)[1].split('\n## Voice, music and typography', 1)[0]
    credits = credits.replace(', new narration and music', '')
    credits = credits.replace('Source credits also appear in the film end card and MP4 metadata.',
                              'Keep this credits file with these editing assets.')
    (OUT / 'CREDITS.md').write_text('# StreetSense product clips — credits\n\n'
        'These silent editing assets show recorded StreetSense outputs and original-frame '
        'Meta evaluations. No new model calls were made. Native overlay code is from '
        'StreetSense (AGPL-3.0). Third-party footage retains its separate license.\n\n'
        + credits)
    (OUT / 'README.md').write_text(
        '# StreetSense product video assets\n\n'
        'Twelve silent 1280x720 H.264 clips at 15 fps. No promo narration, '
        'music, intro/outro or 3D. The combined reel joins the clips with straight cuts.\n\n'
        'Clips 01-07 redraw the native product overlay from actual saved frame-by-frame '
        'detections, track IDs, local readings and selected cue text. They are recorded '
        'offline product outputs, not live UI captures or field trials. Meta was disabled '
        'for those runs. Unrecorded surface/depth overlays are omitted.\n\n'
        'Clips 08-11 are labelled five-second holds on actual Muse evaluation input '
        'frames with the recorded validated results and production crossing cue. '
        'They are not continuous live model responses. Clip 12 reveals an actual '
        'SAM crosswalk mask with an editorial wipe.\n\n'
        'Preserve these evidence labels. GO requires local WALK and validated Muse YES. '
        'Unclear means wait; maps, masks and other models never authorize crossing. '
        'Read CREDITS.md for footage attribution and reuse conditions. '
        'manifest.json includes durations, provenance and exact selected cue text.\n'
    )
    (OUT / 'concat.txt').write_text(''.join(f"file '{m['file']}'\n" for m in manifest))
    subprocess.run([FF, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i',
                    str(OUT / 'concat.txt'), '-c', 'copy', '-movflags', '+faststart',
                    str(OUT / 'StreetSense-product-only-reel.mp4')], check=True)
    print('Product clips and combined reel ready', flush=True)


if __name__ == '__main__':
    main()
