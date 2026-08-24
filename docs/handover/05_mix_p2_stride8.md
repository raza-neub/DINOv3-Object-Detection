# Experiment 5: Mix P2 Stride-8 + Phase C (ViT-S/16+)

## Status: COMPLETED (ViT-S+ runs), last 3 runs DROPPED due to low scores

## Summary

Added 4th FPN level (P2, stride 8) for small objects, plus "Phase C" architectural improvements: DFL regression, DCNv2 in cls tower, AIFI self-attention on P5, auxiliary decoder supervision. Warm-started from mix `best_e250.pth`. Also introduced small-object augmentations (copy-paste, SAHI-crop) and reduced hue jitter.

**This was the best mAP result (0.6262) but the last 3 ViT-S+ runs showed low scores and evaluation plateau, motivating the switch to ConvNeXt.**

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `config/config_mix_p2.py` | P2 + Phase C, 250 epochs from-scratch or warm-start |
| Backbone | `src/model_backbone_v3.py` | `DinoBackboneV3`: 3 depths [3, 6, 11] from ViT-S/16+ |
| Head | `src/model_head_mix_p2.py` | `DINODetectionHeadMixP2`: 4-level [P2,P3,P4,P5], dual cls |
| Loss | `src/loss_v3.py` | VFL + DFL + STAL + optional AuxDecoder loss |
| Utils | `src/utils_OBB.py` | OBB decode, traffic-light cross-class NMS |
| Dataset | `src/dataset_coco_v3.py` | Mosaic + copy-paste + SAHI-crop + flip + translate |
| Training | `train/train_mix_p2.py` | 4-stage curriculum with warm-start support |
| Eval | `eval_mix_p2.py` | Full COCO-style AP (10 metrics) |
| Inference | `inference/inference_batch_v2.py` | Auto-detects mix_p2 head |

## Architecture

```
Input (480x640) --> DinoBackboneV3 (ViT-S/16+, frozen, 384-d)
  --> [shallow, mid, deep] each (B, 384, 30, 40)
  --> MultiLevelEfficientPANP2 (4-level):
      P5 (stride 64, 8x10):  downsample(deep_proj)
      P4 (stride 32, 15x20): downsample(mid_proj)
      P3 (stride 16, 30x40): shallow_proj + top-down/bottom-up
      P2 (stride 8, 60x80):  upsample(P3) + SmallObjectRefine  <-- NEW
  --> [Optional] AIFI: multi-head self-attention on P5 (2 layers, 8 heads)
  --> DetectHeadMix (num_levels=4):
      cls_tower: 6 RepDWSBlock @ 256ch (with DCNv2 in last 2 blocks)
      reg_tower: 4 RepDWSBlock @ 256ch
      cls_neubie[L]: CosineConv2d, 16 classes (per-level)
      cls_coco[L]:   CosineConv2d, 80 classes (per-level)
      bbox_reg[L]:   4*(reg_max+1) = 68 channels (DFL)  <-- NEW
  --> [Optional] AuxDecoderHead: 100 queries, 2 layers, Hungarian matching
```

## Key Changes vs Mix (config_mix.py)

| Parameter | mix | mix_p2 | Reason |
|-----------|-----|--------|--------|
| FPN levels | 3 (P3/P4/P5) | **4 (P2/P3/P4/P5)** | Stride-8 for small objects |
| FPN_CH | 192 | **256** | More capacity for 4 levels |
| TAL_TOPK | 12 | **16** | More anchors with P2's 4x increase |
| BATCH_SIZE | 128 | **64** | STAL anchor 4x memory |
| MOSAIC_PROB | 0.5 | **0.3** | Reduced — mosaic shrinks objects |
| HUE_DELTA | 10 | **2** | Hue IS the label for traffic lights |
| SATURATION_RANGE | (0.8, 1.2) | **(0.9, 1.1)** | Conservative |
| COPYPASTE_PROB | n/a | **0.5** | Paste rare classes |
| SAHICROP_PROB | n/a | **0.25** | Mirror inference SAHI geometry |
| ZOOMOUT_PROB | n/a | **0.0** | DROPPED: shrinks tiny TL below trainable |
| USE_AIFI | n/a | **True** | Self-attention on P5 |
| USE_DFL | n/a | **True** | Distribution Focal Loss (reg_max=16) |
| USE_CENTERNESS | True | **False** | VFL already IoU-aware |
| USE_DCN | n/a | **True** | DCNv2 in cls tower |
| USE_AUX_DECODER | n/a | **True** | Hungarian-matched aux supervision |
| VFL_Q_FLOOR | n/a | **0.3** | Min positive weight for rare classes |
| WARM_START_CKPT | n/a | **best_e250.pth** | 3-level -> 4-level partial load |

## Small-Object Augmentations (NEW)

```python
# Copy-paste: duplicate rare/small classes into training images
COPYPASTE_PROB        = 0.5
COPYPASTE_MAX_PER_IMG = 5
COPYPASTE_CLASSES     = ('bollard', 'scooter', 'warning_light',
                         'traffic_light_red', 'traffic_light_green',
                         'traffic_light_other')
COPYPASTE_SCALE_RANGE = (0.6, 1.4)

# SAHI-crop: upper-strip tile upsampled to IMG_SIZE
# Mirrors inference SAHI geometry so training matches magnified distribution
SAHICROP_PROB       = 0.25
SAHICROP_UPPER_FRAC = 0.55
SAHICROP_COLS       = 2
SAHICROP_OVERLAP    = 0.2
SAHICROP_MIN_VIS    = 0.3

# ZoomOut: DROPPED (shrinks 8px traffic lights to ~3px = unlearnable)
ZOOMOUT_PROB = 0.0
```

## Phase C Architectural Improvements

### C1: AIFI (Attention-based Intra-scale Feature Interaction)
- Multi-head self-attention on P5 (coarsest level)
- 2 layers, 8 heads, 2D sinusoidal positional encoding
- Only 80 tokens for 480x640 input — low overhead
- Helps global context for large objects

### C2: DFL (Distribution Focal Loss)
- Head outputs `4 * (reg_max+1)` channels per box side (4 * 17 = 68)
- Soft two-bin cross-entropy from continuous GT offset
- 16x more localization expressiveness; models uncertainty
- Replaces direct LTRB softplus regression

### C3: Centerness Disabled
- VFL already provides IoU-aware quality scores
- Centerness branch removed to reduce head complexity

### C4: DCNv2 in Classification Tower
- Deformable Convolution v2 in last 2 cls tower blocks
- Better spatial adaptation for irregular object shapes

### C5: Auxiliary Decoder (training only)
- Lightweight transformer decoder on FPN features
- 100 learnable object queries, 2 layers
- Hungarian matching between queries and GT boxes
- Helps stabilize early training; discarded at inference

## Warm-Start Mechanism

```python
WARM_START_CKPT = 'results/mixed-dataset-training/mix_best_20260604/best_e250.pth'
```

- Old checkpoint: 3-level (P3/P4/P5), FPN_CH=192
- New model: 4-level (P2/P3/P4/P5), FPN_CH=256
- **Shape-filtered loading**:
  - cls_tower, cosine classifiers (cls_channels=256 matches) -> warm-start
  - FPN projectors (384->256 vs 384->192) -> fresh init (shape mismatch)
  - bbox_reg (DFL), DCNv2, AIFI, aux_decoder, P2 modules -> fresh init
- Old level mapping: level i (P3/P4/P5) -> new level i+1; new level 0 (P2) cloned from old level 0 (P3)

## Backbone Unfreeze Option (Phase 3)

```python
UNFREEZE_LAST_N = 0     # Phase 2: frozen (default)
# UNFREEZE_LAST_N = 2   # Phase 3: unfreeze blocks 10-11
LR_BACKBONE     = 1e-5  # ~LR_SHARED/20
BACKBONE_WD     = 1e-4  # weights only; norm/bias get 0
GRAD_CLIP_NORM  = 0.1   # tightened when backbone unfrozen (from 5.0)
```

Unfreeze is **stage-gated**: only in Stage 3 (mixed COCO+Neubie), re-frozen in Stage 4 (calibration).

## Training Commands

```bash
# From-scratch (full 4-stage, 250 epochs):
nohup python -u train/train_mix_p2.py --config config/config_mix_p2.py \
  --use-amp > nohup_mix_p2.out 2>&1 &

# Warm-start from 3-level best_e250:
nohup python -u train/train_mix_p2.py --config config/config_mix_p2.py \
  --warm-start results/mixed-dataset-training/mix_best_20260604/best_e250.pth \
  --use-amp > nohup_mix_p2_warm.out 2>&1 &

# Resume training:
nohup python -u train/train_mix_p2.py --config config/config_mix_p2.py \
  --resume results/mixed-dataset-training/<run>/last.pth \
  --use-amp > nohup_mix_p2_resume.out 2>&1 &

# With backbone unfreeze (Phase 3):
nohup python -u train/train_mix_p2.py --config config/config_mix_p2.py \
  --unfreeze-last-n 2 --use-amp > nohup_mix_p2_unfreeze.out 2>&1 &
```

## Evaluation

```bash
# Full COCO-style AP (10 metrics):
python eval_mix_p2.py --config config/config_mix_p2.py \
  --checkpoint results/mixed-dataset-training/<run>/best.pth \
  --device cuda:0 --split val

# Per-class threshold calibration:
python calibrate_thresholds.py --config config/config_mix_p2.py \
  --checkpoint results/mixed-dataset-training/<run>/best.pth \
  --model-type mix --device cuda:0

# SAHI effectiveness A/B test:
python sahi_tl_eval.py --config config/config_mix_p2.py \
  --checkpoint results/mixed-dataset-training/<run>/best.pth \
  --device cuda:3 --max-images 1500
```

## Training Runs

### mix_p2_20260616 (warm-start, 80 epochs, P2 proof-of-concept)
- Warm-started from best_e182.pth
- Reached epoch 7 only (early experiment)
- val_macro_f1: 0.632 at epoch 7

### mix_p2_v199_scratch_20260618 (from-scratch, 199 epochs)
- No warm-start, full 4-stage curriculum
- Explored from-scratch P2 training

### mix_p2_v199_warmstart_20260623 (warm-start, 199 epochs)
- Warm-started from best_e250.pth
- Compared warm-start vs from-scratch

### mix_p2_256ch_e182_20260623 (256ch, warm-start)
- FPN_CH=256 (up from 192)
- Warm-start with shape-filtered loading

### mix_p2_256ch_240x320_20260630 (lower resolution experiment)
- Tested 240x320 input (half resolution)

### mix_p2_vitl (ViT-L backbone experiment)
- Switched to larger ViT-L/16 backbone (1024-d embed, 24 layers)
- `nohup_mix_p2_vitl_resume_20260731.out` (202 MB) — reached epoch 246/250

## Best Results (mix_p2 best_e080)

```
mAP @ IoU=0.50:      0.6262
mAP @ IoU=0.50:0.95: 0.3915

Per-class AP@50:
  ev_close:            0.87
  ev_open:             0.82
  neubie:              0.81
  car:                 0.76
  bus:                 0.74
  warning_light:       0.74
  traffic_light_red:   0.56
  traffic_light_green: 0.54
  traffic_light_other: 0.37
  bollard:             0.44
  scooter:             0.29
```

## Why the Last 3 ViT-S+ Runs Were Dropped

1. **Low scores**: The final 3 ViT-S+ runs (mix_p2_256ch variants) showed diminishing returns. mAP and F1 plateaued despite hyperparameter tuning.
2. **Evaluation plateau**: Per-class metrics for small objects (scooter, bollard, traffic lights) stopped improving. The ViT-S+ backbone at stride 16 fundamentally cannot resolve very small objects well — all intermediate taps are at the same 30x40 resolution, and P2 is synthesized by upsampling (not native stride-8 features).
3. **Inference speed**: ViT-S+ at 480x640 runs at ~3.7 img/s in evaluation — far below the 25-40 FPS target for edge deployment.
4. **Decision**: Switch to ConvNeXt backbone which provides native multi-scale features (stride 8/16/32 built-in), faster inference (CNN vs attention), and higher performance ceiling.

## Files Reference

```
config/config_mix_p2.py                       # Config
train/train_mix_p2.py                         # Training script
eval_mix_p2.py                                # Evaluation (COCO-style 10 metrics)
src/model_backbone_v3.py                      # Backbone
src/model_head_mix_p2.py                      # Detection head (4-level, dual cls)
src/loss_v3.py                                # Loss (VFL + DFL + AuxDecoder)
src/utils_OBB.py                              # Decode + NMS
src/dataset_coco_v3.py                        # Dataset (mosaic + copy-paste + SAHI-crop)
src/class_names.txt                           # 16 Neubie classes
inference/inference_batch_v2.py               # Batch video inference
calibrate_thresholds.py                       # Per-class threshold tuning
sahi_tl_eval.py                               # SAHI A/B test
results/mixed-dataset-training/mix_p2_*/      # Checkpoints
docs/model_head_mix_p2_pipeline.md            # Full architecture reference
docs/unfreeze_backbone_edgecrafter_design.md  # Backbone unfreeze design doc
```
