# Experiment 6: ConvNeXt Backbone (ConvNeXt-Small / ConvNeXt-Base)

## Status: ACTIVE EXPERIMENTATION

## Summary

Switched from ViT-S/16+ to DINOv3 ConvNeXt backbone for two reasons:
1. **Fast FPS**: ConvNeXt is a pure CNN — no self-attention, faster inference than ViT, better suited for edge deployment (Jetson Orin via TensorRT).
2. **High performance**: Native multi-scale features at stride 8/16/32 (no artificial resampling like ViT), with different channel dimensions per stage. P2 is a real stride-8 feature map, not an upsample artifact.

The last 3 ViT-S+ mix_p2 runs showed score plateaus. ConvNeXt offers a higher performance ceiling.

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config (Small) | `config/config_convnext_small.py` | ConvNeXt-Small, 180 epochs, BS=112 |
| Config (Base) | `config/config_convnext_base.py` | ConvNeXt-Base, 180 epochs, BS=80 |
| Backbone | `src/model_backbone_convnext.py` | `DinoBackboneConvNeXt`: 3 stages at stride 8/16/32 |
| Head | `src/model_head_convnext.py` | `ConvNeXtDetectionHeadMixP2`: ConvNeXt-aware FPN + mix head |
| Loss | `src/loss_v3.py` | VFL + DFL + STAL + AuxDecoder (same as mix_p2) |
| Utils | `src/utils_OBB.py` | OBB decode, traffic-light cross-class NMS |
| Dataset | `src/dataset_coco_v3.py` | Mosaic + copy-paste + SAHI-crop |
| Training | `train/train_convnext.py` | 4-stage curriculum (fork of train_mix_p2.py) |
| Eval | `eval_convnext.py` | Full COCO-style AP (same as eval_mix_p2.py) |
| Inference | `inference/inference_batch_v2.py` | Auto-detects ConvNeXt via `fpn.proj_s8.0.weight` key |

## Architecture: Why ConvNeXt is Different

### ViT Backbone (old)
```
All intermediate taps at SAME resolution (30x40) and SAME channels (384):
  block 3 (shallow) --> (B, 384, 30, 40)   stride 16
  block 6 (mid)     --> (B, 384, 30, 40)   stride 16
  block 11 (deep)   --> (B, 384, 30, 40)   stride 16
P2 must be UPSAMPLED from stride-16 features --> NOT true stride-8
```

### ConvNeXt Backbone (new)
```
Stages at DIFFERENT resolutions AND different channels:
  stage 0 (stride 4):  SKIPPED (too fine-grained for detection)
  stage 1 (stride 8):  (B, C1, 60, 80)    <-- NATIVE stride-8 (real P2!)
  stage 2 (stride 16): (B, C2, 30, 40)
  stage 3 (stride 32): (B, C3, 15, 20)
Only P5 (stride 64) is synthesized by downsampling P4
```

### Pyramid Mapping (480x640 input)

| Level | Stride | Feature Map | Source | Notes |
|-------|--------|-------------|--------|-------|
| P2 | 8 | 60x80 | stage 1 (C1) | **Native stride-8** (not upsampled!) |
| P3 | 16 | 30x40 | stage 2 (C2) | Native stride-16 |
| P4 | 32 | 15x20 | stage 3 (C3) | Native stride-32 |
| P5 | 64 | 8x10 | downsample(P4) | Synthesized (only artificial level) |

### Channel Dimensions

| Backbone | Stage 1 (C1) | Stage 2 (C2) | Stage 3 (C3) |
|----------|-------------|-------------|-------------|
| ConvNeXt-Tiny | 96 | 192 | 384 |
| ConvNeXt-Small | 192 | 384 | 768 |
| ConvNeXt-Base | 256 | 512 | 1024 |
| ConvNeXt-Large | 384 | 768 | 1536 |

## Backbone Detail (`src/model_backbone_convnext.py`)

```python
class DinoBackboneConvNeXt(nn.Module):
    def __init__(self, dino_model, dino_weights):
        # Load DINOv3 ConvNeXt model
        # Returns stages 1, 2, 3 (skips stage 0 at stride 4)

    def forward(self, x):
        features = self.backbone.get_intermediate_layers(x, n=[1, 2, 3], reshape=True)
        return list(features)  # [(B,C1,60,80), (B,C2,30,40), (B,C3,15,20)]
```

## Head Detail (`src/model_head_convnext.py`)

### ConvNeXtEfficientPAN (FPN)
```
Unlike ViT FPN (which uses same-shape projections for all levels):
  - Per-stage projection layers with DIFFERENT input channels:
    proj_s8:  Conv(C1 -> FPN_CH)   for stage 1
    proj_s16: Conv(C2 -> FPN_CH)   for stage 2
    proj_s32: Conv(C3 -> FPN_CH)   for stage 3
  - NO artificial downsampling for P2/P3/P4
  - Only P5 = strided_conv(P4, stride=2)
  - Top-down + bottom-up smoothing with RepDWSBlock
```

### Optional AIFI (DISABLED for ConvNeXt)
```python
USE_AIFI = False    # ConvNeXt empirically benefits less from AIFI
```
AIFI is disabled per ablation plan: ConvNeXt already has strong local features from convolutions; the self-attention on P5 that helps ViT is less impactful here.

### Optional AuxDecoderHead (ENABLED)
- Transformer decoder for training supervision (100 queries, 2 layers)
- Hungarian matching, discarded at inference

## Key Config Differences

| Parameter | mix_p2 (ViT) | convnext_small | convnext_base |
|-----------|---------------|----------------|---------------|
| BACKBONE_TYPE | (implicit ViT) | **'convnext'** | **'convnext'** |
| DINO_MODEL | dinov3_vits16plus | **dinov3_convnext_small** | **dinov3_convnext_base** |
| CONVNEXT_IN_CHANNELS | n/a | **[192, 384, 768]** | **[256, 512, 1024]** |
| USE_AIFI | True | **False** | **False** |
| WARM_START_CKPT | best_e250.pth | **''** (from scratch) | **''** (from scratch) |
| NUM_EPOCHS | 250 | **180** | **180** |
| BATCH_SIZE | 64 | **112** | **80** |
| STAGE1_END | 50 | **36** | **36** |
| STAGE2_END | 70 | **50** | **50** |
| STAGE3_END | 230 | **166** | **166** |
| STAGE3_SUB1_END | 110 | **79** | **79** |
| STAGE3_SUB2_END | 145 | **104** | **104** |
| VAL_METRIC_EVERY | 2 | **10** | **10** |

Everything else (loss, augmentation, curriculum structure, LRs) is identical to mix_p2.

## Pretrained Backbone Weights

Stored in `checkpoints/`:
```
dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth    (107 MB)
dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth   (192 MB)
dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth    (338 MB)
dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth   (781 MB)
```

## Training Commands

```bash
# ConvNeXt Small (from scratch, 180 epochs):
nohup python -u train/train_convnext.py \
  --config config/config_convnext_small.py \
  --use-amp > nohup_convnext_small.out 2>&1 &

# ConvNeXt Base (from scratch, 180 epochs):
nohup python -u train/train_convnext.py \
  --config config/config_convnext_base.py \
  --use-amp --device cuda:1 > nohup_convnext_base.out 2>&1 &

# Resume:
nohup python -u train/train_convnext.py \
  --config config/config_convnext_small.py \
  --resume results/mixed-dataset-training/<run>/last.pth \
  --use-amp > nohup_convnext_small_resume.out 2>&1 &
```

## Evaluation Commands

```bash
# Full COCO-style AP:
python eval_convnext.py --config config/config_convnext_small.py \
  --checkpoint results/mixed-dataset-training/<run>/best.pth \
  --device cuda:0 --split val

# Calibration:
python calibrate_thresholds.py --config config/config_convnext_small.py \
  --checkpoint results/mixed-dataset-training/<run>/best.pth \
  --model-type convnext --device cuda:0
```

## Batch Inference

```bash
# Standard inference:
python inference/inference_batch_v2.py \
  --config config/config_convnext_small.py \
  --device cuda:0 --tile 3 --aabb --show-fps

# With SAHI tiling (for traffic lights):
python inference/inference_batch_v2.py \
  --config config/config_convnext_small.py \
  --device cuda:0 --tile 3 --aabb --show-fps --sahi
```

Visualization outputs: `visualization/convnext_small/`, `visualization/convnext_base_sahi/`

## Training Runs

### ConvNeXt-Small Run 1: 2026-08-12 (initial, 17 epochs)
- From scratch, BATCH_SIZE=112, NUM_EPOCHS=250
- Stopped early (GPU issues / restart)
- train_loss: 1.980, val_loss: 2.250 (Stage 1 — COCO warmup)

### ConvNeXt-Small Run 2: 2026-08-13 (resumed, 180 epochs)
```bash
nohup python -u train/train_convnext.py \
  --config config/config_convnext_small.py \
  --resume results/mixed-dataset-training/<run1>/last.pth \
  --use-amp --batch-size 104 > nohup_convnext_small.out 2>&1 &
```
- Resumed from run 1, NUM_EPOCHS=180, compile_backbone=True
- **Reached epoch 179**: train_loss=1.481, val_loss=1.318
- **val_macro_f1: 0.435, val_crucial_f1: 0.244**
- At epoch 175: Stage 4 (Neubie calibration)

### ConvNeXt-Small Run 3: 2026-08-21 (calibration mode)
- Resumed from run 2, calibration-specific settings
- CLI_CALIBRATE=true, CLI_CALIBRATE_EPOCHS=20, CLI_CALIBRATE_LR=0.0005
- **Reached epoch 180**: train_loss=0.535, val_loss=0.558
- **val_macro_f1: 0.431, val_crucial_f1: 0.240**
- Calibrated F1 = 0.430958

### ConvNeXt-Base: 2026-08-13 (in progress)
- From scratch, BATCH_SIZE=80, NUM_EPOCHS=180
- **Currently at epoch 157/180** (Stage 3 — mixed training)
- Batch rate: ~15.51 s/it (slower than Small due to larger model)
- LRs: shared=4.96e-06, cls_coco=2.09e-05, cls_neubie=2.09e-05

## Current Results Summary

| Run | Epochs | macro-F1 | crucial-F1 | Status |
|-----|--------|----------|------------|--------|
| ConvNeXt-Small (calibrated) | 180 | 0.431 | 0.240 | Completed |
| ConvNeXt-Base | 157/180 | — | — | In progress |

Note: F1 scores are lower than mix (0.6919) because ConvNeXt is training from scratch (no warm-start possible — different backbone), and the 180-epoch compressed schedule is shorter than mix's 250 epochs. Performance is expected to improve with:
- Full training completion
- Larger batch sizes
- Potential backbone unfreeze (ConvNeXt stages)
- AIFI re-evaluation

## Differences in Training Script (`train/train_convnext.py`)

Fork of `train/train_mix_p2.py` with:
- Imports `DinoBackboneConvNeXt` instead of `DinoBackboneV3`
- Imports `ConvNeXtDetectionHeadMixP2` instead of `DINODetectionHeadMixP2`
- Model construction uses `config.CONVNEXT_IN_CHANNELS` (per-stage list) instead of embed_dim
- **No warm-start logic** (training from scratch with new backbone)
- **No backbone unfreeze support** (ConvNeXt uses stages, not ViT blocks — requires different logic)
- Everything else identical (curriculum, augmentation, loss, optimizer)

## Next Steps / Open Questions

1. **ConvNeXt-Base completion**: Wait for epoch 180 completion, run full COCO-style eval
2. **ConvNeXt-Small vs Base**: Compare mAP and FPS — is Base worth the 2x size?
3. **ConvNeXt-Large**: checkpoints/dinov3_convnext_large (781 MB) is available but untested
4. **AIFI re-evaluation**: Currently disabled for ConvNeXt — worth A/B testing
5. **Backbone unfreeze for ConvNeXt**: Different mechanism needed (stage-based, not block-based)
6. **TensorRT export**: Need to rebuild export pipeline (inference/export_model.py was deleted)
7. **FPS benchmarking**: Formal latency comparison ConvNeXt vs ViT on Jetson Orin

## Files Reference

```
config/config_convnext_small.py               # Config (Small)
config/config_convnext_base.py                # Config (Base)
train/train_convnext.py                       # Training script
eval_convnext.py                              # Evaluation
src/model_backbone_convnext.py                # ConvNeXt backbone wrapper
src/model_head_convnext.py                    # ConvNeXt-aware FPN + mix head
src/loss_v3.py                                # Loss (shared with mix_p2)
src/utils_OBB.py                              # Decode + NMS (shared)
src/dataset_coco_v3.py                        # Dataset (shared)
src/class_names.txt                           # 16 Neubie classes
inference/inference_batch_v2.py               # Batch video inference
calibrate_thresholds.py                       # Per-class threshold tuning
checkpoints/dinov3_convnext_*.pth             # Pretrained backbone weights
results/mixed-dataset-training/*convnext*/    # Training run outputs
visualization/convnext_small/                 # Inference visualizations
visualization/convnext_base_sahi/             # SAHI inference visualizations
nohup_convnext_small.out                      # Training log (48 MB)
nohup_convnext_base.out                       # Training log (41 MB)
```
