"""Fine-tune YOLO11s on the pedsig set (scripts/pedsig/build_yolo_ds.py). CPU, modest.

Every epoch is saved (save_period=1) so the checkpoint can be picked on PTLD-valid
signal recall with scripts/pedsig/pick_checkpoint.py (validation only, never test).
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/datasets/pedsig/yolo/data.yaml")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--freeze", type=int, default=10)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--name", default="yolo11s_pedsig")
a = ap.parse_args()
YOLO("models/yolo11s.pt").train(
    data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, device="cpu", workers=4,
    freeze=a.freeze, optimizer="AdamW", lr0=0.0005, cos_lr=True, warmup_epochs=1,
    close_mosaic=2, save_period=1, patience=100, seed=0, deterministic=False,
    project="/workspace/sr_logs/pedsig_yolo", name=a.name, exist_ok=True, plots=False, amp=False)
