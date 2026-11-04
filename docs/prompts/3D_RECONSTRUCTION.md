# Prompt: continue the isolated 3D experiment

```text
Work in StreetSense. Continue experiments/reconstruction as a separate offline module to determine whether 3D reconstruction improves pedestrian scene understanding. Read AGENTS.md, experiments/reconstruction/README.md, docs/RECONSTRUCTION_EXPERIMENT.md, the saved reports, and the existing depth, tracking and segmentation code first.

Preserve the crossing rule: local pedestrian reader WALK AND validated Muse Spark YES are both required. Reconstruction, maps, camera pose or distance can never authorize crossing or refresh signal evidence. Keep the experiment outside the production pipeline. Do not deploy or enable it by default.

The module already runs the Apache-licensed Meta MapAnything checkpoint, exports cameras/depth/point clouds, compares the existing relative-depth output under shared cameras, rejects bad sequences, and generates an orbitable demo. Reproduce the recorded moving-camera and static controls before claiming an improvement. Retain previous reports. Do not mistake a plausible point cloud or low reprojection error under little parallax for accurate reconstruction.

Next, build a small evaluation set with calibrated intrinsics and independently measured distances, curb heights, obstacles and camera motion. Include walking camera motion, stops, pure rotation, low light, motion blur, low texture, moving people/vehicles and occlusions. Record source licenses, frame timestamps and hashes. Keep development and held-out scenes separate.

Compare: existing pinhole distance priors, existing calibrated Depth Anything V2 Small, MapAnything, and an optional independently estimated geometric pose baseline. First test dynamic-object exclusion using existing detection/tracking/segmentation. If adding pose refinement or bundle adjustment, fit on one set of correspondences and evaluate on disjoint tracks/views. Do not evaluate on the points used to fit cameras. Test improvements independently and retain failed cases.

Measure metric depth and obstacle-distance error, curb-height error, free-space false negatives, static-surface consistency, pose drift, valid coverage, cold load time, warm median/p95 latency, process peak RSS and accelerator memory. Report sample counts, hardware, resolution, precision, versions, model revisions and synchronization. Do not call scale alignment to another learned model ground truth. State clearly when a metric cannot be measured.

Keep dependencies and caches separate using the existing setup/run scripts. Prefer the Apache MapAnything checkpoint. Research current official Meta implementations and licensing before adding another model. Avoid adding a generative object-completion model as evidence of observed pavement or hidden obstacles.

Generate a small demo from real reconstruction output: source frames, colored point cloud, camera path, baseline comparison and visible rejection states. Include at least one failure control. Verify the viewer's controls in a browser, run meaningful geometry/isolation tests, and give a concrete recommendation: continue offline research, try an asynchronous expiring advisory layer, or stop because the costs outweigh the measured gain. Do not integrate with navigation without a separately reviewed change and calibrated evidence.

Deliver working code, reproducible commands, measured results with limitations, the demo, and the next experiment most likely to resolve the remaining uncertainty. Make useful progress without waiting for optional preferences.
```
