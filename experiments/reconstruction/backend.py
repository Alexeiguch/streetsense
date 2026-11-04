"""Optional Meta model adapter. Heavy imports occur only on explicit invocation."""
from __future__ import annotations

import time
import hashlib
import importlib.metadata
import subprocess
import os
from pathlib import Path

MODEL_ID = "facebook/map-anything-apache"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
SOURCE_REVISION = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"


def reconstruct(paths: list[Path], device: str = "auto", size: int = 518, repeats: int = 1):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images
    import mapanything

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple GPU unavailable. Use --device cpu or run with hardware access.")
    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                   allow_patterns=["config.json", "model.safetensors"])
    start = time.perf_counter()
    model = MapAnything.from_pretrained(checkpoint).eval().to(device)
    sync()
    load_s = time.perf_counter() - start
    views = load_images([str(p) for p in paths], resize_mode="longest_side", size=size)
    inference_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.inference_mode():
            predictions = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                      use_amp=False, apply_mask=True, mask_edges=True,
                                      apply_confidence_mask=True, confidence_percentile=10)
        sync()
        inference_times.append(time.perf_counter() - start)
    def array(key):
        return np.stack([p[key][0].detach().cpu().numpy() for p in predictions])
    result = {"points": array("pts3d"), "depth": array("depth_z")[..., 0],
              "cameras": array("camera_poses"), "intrinsics": array("intrinsics"),
              "mask": array("mask")[..., 0].astype(bool), "confidence": array("conf"),
              "rgb": np.clip(array("img_no_norm") * 255, 0, 255).astype(np.uint8)}
    if not np.isfinite(result["cameras"]).all() or not np.isfinite(result["intrinsics"]).all():
        raise ValueError("Model returned non-finite camera geometry")
    # Reject non-finite output even if the learned mask accepts it.
    result["mask"] &= (np.isfinite(result["points"]).all(-1)
                       & np.isfinite(result["depth"]) & (result["depth"] > 0))
    timing = {"device": device, "precision": "float32", "model_load_s": load_s,
              "inference_s": inference_times[-1], "inference_repeats_s": inference_times,
              "views": len(paths), "shape_hw": list(result["depth"].shape[1:]),
              "checkpoint": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
              "source_revision_expected": SOURCE_REVISION, "torch": torch.__version__,
              "mps_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
              "packages": {p: importlib.metadata.version(p) for p in ["mapanything", "uniception", "numpy", "huggingface-hub", "torchvision"]}}
    source_root = Path(mapanything.__file__).resolve().parents[1]
    if (source_root / ".git").is_dir():
        timing["source_revision_actual"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    # UniCeption obtains DINOv2 code through Torch Hub; fingerprint that code too.
    hub_root = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub_root.exists():
        digest = hashlib.sha256()
        for source in sorted(hub_root.rglob("*.py")):
            digest.update(str(source.relative_to(hub_root)).encode())
            digest.update(source.read_bytes())
        timing["dinov2_python_tree_sha256"] = digest.hexdigest()
    if device == "mps":
        timing["mps_driver_allocated_bytes_after_inference"] = torch.mps.driver_allocated_memory()
    del model, predictions
    if device == "mps":
        torch.mps.empty_cache()
    return result, timing
