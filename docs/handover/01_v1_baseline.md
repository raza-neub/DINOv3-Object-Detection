# Experiment 1: v1 Baseline (AABB, ViT-B/16)

## Status: ARCHIVED (files deleted from HEAD)

## Summary

The original baseline detector. PAN-FPN neck with axis-aligned bounding boxes (AABB), trained on COCO only. Used the larger ViT-B/16 backbone (768-d embed, stride 16).

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `config/config.py` | Legacy config (stale paths, 15 epochs, COCO only) |
| Backbone | `src/model_backbone.py` (deleted) | `DinoBackbone`: single output from final ViT block |
| Head | `src/model_head.py` (deleted) | PAN-FPN + shared DWS FCOS towers |
| Loss | `src/loss.py` | TAL + STAL assignment, focal BCE, GIoU reg, BCE centerness, ProgLoss warmup |
| Utils | `src/utils.py` | AABB decode, per-class NMS via torchvision |
| Dataset | `src/dataset_coco.py` | COCO-format loader, photometric-only augmentation |
| Training | `train/train_detector.ipynb` (deleted) | Jupyter notebook trainer |
| Inference | `inference/inference.py` (deleted) | Single-image inference |

## Architecture

```
Input (640x640) --> DinoBackbone (ViT-B/16, frozen, 768-d)
  --> feats[-1] (B, 768, 40, 40)
  --> PAN-FPN (3-level: P3/P4/P5 at strides 16/32/64, 192ch)
  --> Shared DWS towers (4 conv each for cls/reg)
  --> cls (focal), reg (GIoU, LTRB softplus), ctr (BCE)
```

## Key Config Settings

```python
DINO_MODEL     = "dinov3_vits16plus"   # despite name, early runs used vitb16
IMG_SIZE       = 640                    # square input (later versions use 480x640)
FPN_CH         = 192
N_CONVS        = 4
BATCH_SIZE     = 16
LEARNING_RATE  = 0.0001
NUM_EPOCHS     = 15
SCORE_THRESH   = 0.2
NMS_THRESH     = 0.6
```

## Loss Details (`src/loss.py`)

- **STALTaskAlignedAssigner**: Task-Aligned Label assignment
  - STAL expansion: if GT shorter side < `min_stride` (8px), expand candidate region
  - TAL metric: `cls_score^alpha * CIoU^beta` (alpha=0.5, beta=6.0)
  - Top-k per GT, conflict resolution by highest IoU
  - Soft targets: IoU-normalized alignment metrics in [0, 1]
- **Focal BCE**: `alpha*(1-p)^gamma * L` with soft TAL targets
- **GIoU regression**: positives only, `1 - GIoU`
- **Centerness BCE**: LTRB-based centerness target
- **ProgLoss warmup**: linearly ramp reg+ctr weights from 0->1 over first N epochs

## Training Command

```bash
# Was run via Jupyter notebook (deleted)
# Equivalent CLI would be:
python train/train_detector.py --config config/config.py --device cuda:0
```

## Results

No formal COCO mAP evaluation was recorded. This was the initial proof-of-concept.

## Why It Was Dropped

1. **ViT-B/16 too heavy** for edge deployment (Jetson Orin). 768-d embed + large attention = slow inference.
2. **Square 640x640 input** wastes compute on landscape-oriented driving scenes.
3. **COCO-only training** does not learn Neubie-specific classes.
4. **No augmentation pipeline** beyond photometric jitter.
5. Files deleted from HEAD; superseded by v2 with ViT-S/16+ and OBB.

## What Was Kept for Later Versions

- `src/loss.py`: TAL + STAL assignment logic reused as foundation for all later loss variants
- `src/utils.py`: AABB decode/NMS, re-exported by `utils_OBB.py` for backward compatibility
- `src/dataset_coco.py`: COCO-format dataset loader (v1/v2 share it)
