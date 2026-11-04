"""Paths for both a portable source bundle and scripts/submission_demo in the repo."""
import os,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT_ROOT=Path(os.environ.get('STREETSENSE_ROOT',next((str(p) for p in HERE.parents if (p/'pyproject.toml').exists() and (p/'streetsense').is_dir()),str(HERE.parent.parent.parent))))
ROOT=Path(os.environ.get('STREETSENSE_DEMO_ROOT',str(HERE.parent if HERE.name=='source' else PROJECT_ROOT/'outputs/submission-demo')))
FF=os.environ.get('FFMPEG') or shutil.which('ffmpeg') or '/private/tmp/streetsense-video-tools/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
