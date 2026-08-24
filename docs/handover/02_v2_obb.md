# Experiment 2: v2 OBB (Oriented Bounding Boxes, ViT-S/16+)

## Status: COMPLETED, ARCHIVED (2 training runs)

## Summary

Switched to lighter ViT-S/16+ backbone (384-d embed). Added Oriented Bounding Box (OBB) support with ProBIoU-based assignment and sin^2(2*delta_theta) angle loss. Edge-optimized head with RepDWS blocks and ECA attention.

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `config/config.py` (legacy, modified per-run) | Per-run overrides via CLI/run_config.json |
| Backbone | `src/model_backbone.py` (deleted from HEAD) | `DinoBackbone`: single output from final ViT block, 384-d |
| Head | `src/model_head_v2.py` | EfficientPAN (bidirectional), RepDWSBlock, ECA, OBB angle branch |
| Loss | `src/loss_OBB.py` | TAL+STAL with ProBIoU, sin^2(2*dtheta) angle loss, vectorized topk |
| Utils | `src/utils_OBB.py` | ProBIoU, rotated NMS, dist2rbox, OBB decode |
| Dataset | `src/dataset_coco.py` | COCO-format loader (shared with v1) |
| Optimizer | `src/optimizer.py` | MuSGD (hybrid Muon+SGD) - first version to use it |
| Inference | `inference/inference.py` (deleted) | Single-image OBB inference |

## Architecture

```
Input (480x640) --> DinoBackbone (ViT-S/16+, frozen, 384-d)
  --> feats[-1] (B, 384, 30, 40)
  --> EfficientPAN (bidirectional, 3-level: P3/P4/P5 at strides 16/32/64, 192ch)
  --> RepDWSBlock towers (4x cls, 4x reg, shared across levels)
  --> SmallObjectRefine on P3
  --> ECA channel attention
  --> cls (focal), reg (GIoU, LTRB softplus), ctr (BCE), angle (sigmoid-scaled)
```

## Key Improvements Over v1

1. **Backbone**: ViT-B/16 (768-d) -> ViT-S/16+ (384-d) = 2x lighter, better for edge
2. **Input**: 640x640 square -> 480x640 landscape (matches driving cameras)
3. **FPN**: Unidirectional -> Bidirectional (top-down + bottom-up)
4. **Blocks**: Standard conv -> RepDWSBlock (reparameterizable, depthwise-separable)
5. **OBB**: Axis-aligned -> Oriented boxes with angle prediction
6. **Angle loss**: sin^2(2*dtheta) handles pi/2 periodicity correctly (smooth_L1 fails at boundaries)
7. **STAL**: Adaptive expansion (tiny: 4x expand + topk+8; small: 3x + topk+4; normal: 2x + topk=10)
8. **Vectorized**: _topk_mask() fully vectorized (no Python for-loops / GPU-blocking .item() calls)
9. **OOM guard**: try/except fallback to CPU for STAL computation

## Loss Details (`src/loss_OBB.py`)

```
Components (all normalized by # foreground anchors):
  cls_loss:   focal BCE with soft TAL targets
  reg_loss:   GIoU on positive anchors
  ctr_loss:   BCE centerness
  angle_loss: sin^2(2*delta_theta) * aspect_weight
              - aspect_weight downweights near-square objects (angle ambiguous)
  ProgLoss:   linear warmup of reg+ctr+angle over first N epochs
```

- **ProBIoU alignment** in TAL metric: replaces CIoU when angles available (rotation-aware)
- **Stride-aware L1**: normalize by per-anchor stride so P3 and P5 contribute equally

## Optimizer (`src/optimizer.py` — MuSGD)

First pipeline to use the MuSGD hybrid optimizer:
- **Muon component**: Newton-Schulz orthogonalization (5 iters) for 2D+ tensors (conv/linear)
- **SGD component**: classical momentum for 1D tensors (bias, GroupNorm)
- Weight decay only on 2D+ tensors
- `lr=3e-4, momentum=0.95, nesterov=True`

## Training Runs

### Run 1: 2026-04-22 (120 epochs)
```bash
nohup python -u train/train_v2.py --config config/config.py \
  --device cuda:3 --batch-size 64 --epochs 120 --lr 3e-4 --use-amp \
  > nohup_v2_run1.out 2>&1 &
```

**Results** (epoch 119):
```
val_loss: 0.6474 (cls=0.0245, reg=0.4151, ctr=0.2025, angle=0.0257)

Per-class focus metrics @ epoch 119:
  traffic_light_red:   P=0.029  R=0.026  (GT=5497)
  neubie:              P=0.360  R=0.435  (GT=980)
  warning_light:       P=0.000  R=0.000  (GT=2088)
  ev_open:             P=0.086  R=0.269  (GT=193)
  traffic_light_green: P=0.061  R=0.063  (GT=2038)
  traffic_light_other: P=0.073  R=0.052  (GT=12071)
```

**Checkpoint**: `results/2026-04-22_04-31-35_dinov3_vits16plus_v2/` (120 checkpoints)

### Run 2: 2026-04-30 (181 epochs, extended)
```bash
nohup python -u train/train_v2.py --config config/config.py \
  --device cuda:2 --batch-size 64 --epochs 300 --lr 3e-4 --patience 100 --use-amp \
  > nohup_v2_run2.out 2>&1 &
```

**Results** (epoch 180, early-stopped from 300):
```
val_loss: 0.6493 (cls=0.0263, reg=0.4141, ctr=0.2041, angle=0.0245)

Per-class focus metrics @ epoch 180:
  traffic_light_red:   P=0.030  R=0.028  (GT=5497)
  neubie:              P=0.420  R=0.486  (GT=980)
  warning_light:       P=0.000  R=0.000  (GT=2088)
  ev_open:             P=0.100  R=0.249  (GT=193)
  traffic_light_green: P=0.053  R=0.056  (GT=2038)
  traffic_light_other: P=0.073  R=0.054  (GT=12071)
```

**Checkpoint**: `results/2026-04-30_06-58-00_dinov3_vits16plus_v2/` (181 checkpoints, best at epoch 80)

## Why It Was Dropped

1. **OBB angle is pseudo-target**: Ground truth is axis-aligned (COCO format). The angle prediction is just `atan2(h,w)` of aspect ratio, not real object orientation. Evaluation discards rotation. The overhead of angle loss/branch/rotated NMS was not justified.
2. **Single-dataset training**: Only Neubie data, no COCO transfer learning. Under-utilizes available data.
3. **Low per-class scores**: Traffic lights near-zero P/R. warning_light completely missed (P=0, R=0). Fundamental issue is not the head design but the training strategy.
4. **No mosaic/geometric augmentation**: Only photometric jitter. Small objects need stronger augmentation.

## What Was Kept for Later Versions

- `src/model_head_v2.py`: RepDWSBlock, ECA, SmallObjectRefine blocks reused in v3/mix/mix_p2
- `src/loss_OBB.py`: ProBIoU, sin^2(2*dtheta) angle loss available for future OBB work
- `src/utils_OBB.py`: ProBIoU rotated NMS, OBB geometry utilities
- `src/optimizer.py`: MuSGD kept but later replaced by AdamW in mix pipeline (finer per-stage LR control)
