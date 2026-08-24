# Experiment Handover: Overview & Lineage

## Project Summary

Lightweight FCOS-style anchor-free object detection head trained on a **frozen DINOv3 backbone** (Meta ViT / ConvNeXt). Only the detection head trains; the backbone stays frozen. Targets 16 Neubie classes for ROS2 deployment via ONNX/TensorRT.

## Pipeline Lineage

```
v1 (AABB, ViT-B/16)
 |
 v-- dropped: low accuracy, wrong backbone size for edge
 |
v2 (OBB, ViT-S/16+)
 |
 v-- dropped: OBB pseudo-target (GT is axis-aligned), marginal gain, added complexity
 |
v3 (multi-depth FPN + VFL, ViT-S/16+)
 |
 v-- dropped: single-dataset only, no COCO transfer
 |
mix (dual-classifier COCO+Neubie, ViT-S/16+)  <-- best ViT baseline (F1=0.6919)
 |
 v-- plateau: mAP@50=0.53, small-object recall poor
 |
mix_p2 (stride-8 P2 level + Phase C, ViT-S/16+)  <-- mAP@50=0.6262
 |
 v-- dropped last 3 ViT-S+ runs: low scores, evaluation plateau
 |
convnext (ConvNeXt-S/B backbone)  <-- ACTIVE: need fast FPS + high performance
```

## Why Each Transition Happened

| From | To | Reason |
|------|----|--------|
| v1 | v2 | ViT-B/16 too heavy for edge; switched to ViT-S/16+. Added OBB for potential rotation support. |
| v2 | v3 | OBB angle is pseudo-target (GT is axis-aligned COCO format). Overhead not justified. Multi-depth FPN added for better small-object features. |
| v3 | mix | Single-dataset training underutilizes COCO's 80-class diversity. Dual-classifier enables COCO transfer to Neubie. |
| mix | mix_p2 | mAP@50 plateaued at 0.53. Small objects (traffic lights, bollard, scooter) need stride-8 resolution. Phase C adds DFL/DCN/AuxDecoder. |
| mix_p2 (ViT-S+) | convnext | **Last 3 ViT-S+ runs showed low scores and evaluation plateau.** ConvNeXt offers native multi-scale (stride 8/16/32 built-in, no artificial resampling), faster FPS, and higher performance ceiling. |

## Key Metrics Progression

| Version | mAP@50 | mAP@50:95 | F1 (macro) | Status |
|---------|--------|-----------|------------|--------|
| v2 (120 ep) | ~0.35* | ~0.22* | — | Completed, archived |
| v2 (181 ep) | ~0.36* | ~0.23* | — | Completed, archived |
| v3 (120 ep) | ~0.40* | ~0.28* | — | Completed, archived |
| mix (250 ep) | 0.5289 | 0.3346 | 0.6919 | Best ViT baseline |
| mix_p2 (80 ep) | 0.6262 | 0.3915 | — | Completed |
| convnext-small | — | — | 0.435 | Training (ep 180/180) |
| convnext-base | — | — | — | Training (ep 157/180) |

*Estimated from loss curves; no formal COCO-style eval was run on v2/v3.

## Document Index

| File | Pipeline | Contents |
|------|----------|----------|
| [01_v1_baseline.md](01_v1_baseline.md) | v1 | Archived baseline, PAN-FPN + AABB |
| [02_v2_obb.md](02_v2_obb.md) | v2 | OBB experiments, 2 training runs |
| [03_v3_multi_depth.md](03_v3_multi_depth.md) | v3 | Multi-depth FPN + VFL |
| [04_mix_dual_classifier.md](04_mix_dual_classifier.md) | mix | Dual-classifier, 4-stage curriculum, best ViT baseline |
| [05_mix_p2_stride8.md](05_mix_p2_stride8.md) | mix_p2 | P2 stride-8, Phase C improvements |
| [06_convnext.md](06_convnext.md) | convnext | ConvNeXt backbone, active experiments |

## 16 Neubie Classes (fixed order)

```
0: bicycle         1: bus              2: car              3: motorcycle
4: person          5: scooter          6: truck            7: bollard
8: traffic_light_red  9: neubie        10: warning_light   11: ev_close
12: ev_open        13: cart            14: traffic_light_green  15: traffic_light_other
```

## Dataset Paths (machine-specific, update for new machine)

```
NEUBIE_ROOT   = '/media/data/shared/merged_coco'
COCO_ROOT_MIX = '/data2/datasets/coco'
DINOV3_DIR    = '/home/raza/Raza/neubi/object_detection_dinov3/dinov3'
```

## Dependencies

- PyTorch 2.x + CUDA
- DINOv3 submodule: `pip install -e .` from repo root
- OpenCV, NumPy, Matplotlib, pycocotools, torchvision, tqdm
