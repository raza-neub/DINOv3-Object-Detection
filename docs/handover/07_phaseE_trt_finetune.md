# Experiment 7: Phase E Training & TRT Fine-Tuning (Sep 1-28, 2026)

## Status: PHASE E COMPLETE, TRT V2 FINE-TUNE IN PROGRESS

---

## Executive Summary

Phase E represents a head architecture overhaul on top of frozen ConvNeXt backbones, with five design changes aimed at improving rare-class detection and enabling TensorRT INT8 deployment. Training completed with early stopping for both models.

**Key results:**

| Model | Best Epoch | MacroF1 | mAP@50 | mAP@50:95 | Stopped At |
|-------|-----------|---------|--------|-----------|------------|
| ConvNeXt-Small | E110 | 0.4576 | 0.7388 | 0.4810 | E140 (patience=30) |
| ConvNeXt-Base | E90 | 0.4630 | 0.7302 | 0.4714 | E120 (patience=30) |

**What happened this month:**

1. **Sep 1-6**: Finetune checkpoint evaluation, discovery of 3 critical inference bugs (centerness halving, rotated NMS on AABB, DFL double-decode)
2. **Sep 7-8**: VFL Q-Floor=0 scratch training to establish clean baseline (~E50)
3. **Sep 9-11**: Phase E architecture design (5 improvements: Conv2d cls, cross-level attn, DCN, VFL schedule, aux decoder)
4. **Sep 9-21**: Phase E training (resumed from scratch baseline, 4-stage curriculum)
5. **Sep 15-16**: AABB decode function written (fixes all 3 bugs, 13x NMS speedup)
6. **Sep 16**: Deliverable package created for developer handoff
7. **Sep 17-18**: Batch inference evaluation on 9 traffic videos
8. **Sep 21-23**: Phase E training completed (early stopping at E140/E120)
9. **Sep 23-25**: Full COCO-style evaluation on best checkpoints
10. **Sep 28**: TRT-clean head v1 created (`model_head_convnext_trt.py`), fine-tuning launched
11. **Sep 28**: TRT v1 fine-tune completed — F1 dropped from 0.4576 to 0.3044 (BN cold-start problem)
12. **Sep 29**: TRT v2 head improvements: DCN→DilatedConv2d (dilation=2, effective 5×5 RF), BN calibration pass (`calibrate_bn_stats()`), frozen BN fine-tuning (`freeze_bn()`). Added `--calibrate-bn` and `--freeze-bn` flags to training script. V2 fine-tune in progress.

---

## Phase E Architecture Changes

Five changes from the finetune-era head, all flags in `config_convnext_small.py` / `config_convnext_base.py`:

| Change | Config Flag | Why |
|--------|------------|-----|
| Standard Conv2d classifier | `COSINE_CLS = False` | CosineConv2d couples all classes via L2 normalization; Conv2d allows independent per-class learning |
| Cross-level self-attention | `USE_CROSS_LEVEL_ATTN = True` | P2 (small objects) attends to P5 (scene context) for better rare/small class discrimination |
| VFL + TAL alpha schedule | `FOCAL_WARMUP_EPOCHS = 50`, `TAL_ALPHA_SCHEDULE = True` | Hard binary targets for E0-49 stabilizes early training; TAL_ALPHA ramps 0-0.5 over E50-80 |
| Deformable convolution (DCN) | `USE_DCN = True` | Adaptive receptive field for non-rigid/occluded objects |
| Auxiliary DETR decoder | `USE_AUX_DECODER = True` | 100-query 2-layer decoder with Hungarian matching provides dense supervision (Neubie batches only) |

---

## Training Details

### 4-Stage Curriculum (180 epoch schedule)

| Stage | Epochs | What | COCO Ratio |
|-------|--------|------|-----------|
| S1: COCO warmup | 0-36 | COCO-only, build shared features | 100% |
| S2: Neubie warmup | 37-50 | Neubie-only, low shared LR | 0% |
| S3-sub1 | 51-79 | Mixed training | 60% |
| S3-sub2 | 80-104 | Mixed training | 40% |
| S3-sub3 | 105-166 | Mixed training | 20% |
| S4: Neubie polish | 167-180 | Neubie-only, low LR | 0% |

Both models hit early stopping (patience=30) before reaching Stage 4:
- **Small**: Best at E110, stopped at E140 (30 epochs without improvement)
- **Base**: Best at E90, stopped at E120 (30 epochs without improvement)

### Training Configuration

| Parameter | Small | Base |
|-----------|-------|------|
| Backbone | `dinov3_convnext_small` | `dinov3_convnext_base` |
| Channel dims | [192, 384, 768] | [256, 512, 1024] |
| FPN channels | 256 | 256 |
| Batch size | 112 | 96 |
| GPU | cuda:2 | cuda:3 |
| Workers | 12 | 8 |
| LR (shared) | 2e-4 | 2e-4 |
| LR (cls_coco) | 1e-3 | 1e-3 |
| LR (cls_neubie) | 1e-3 | 1e-3 |
| AMP | Yes | Yes |
| EMA | Yes (decay=0.999) | Yes (decay=0.999) |

### MacroF1 Learning Curve (Small)

| Epoch | MacroF1 | Stage | Notes |
|-------|---------|-------|-------|
| 9 | 0.0000 | S1 | No Neubie evaluation |
| 39 | 0.1296 | S2 | Neubie classifier starting |
| 49 | 0.1313 | S2 | End of warmup |
| 59 | 0.3398 | S3-sub1 | Large jump from mixed training |
| 79 | 0.4433 | S3-sub1 | Approaching finetune level |
| 99 | 0.4533 | S3-sub2 | Near finetune |
| 109 | 0.4576 | S3-sub2 | **Best -- surpasses all previous** |
| 119 | 0.4564 | S3-sub3 | Slight dip (COCO ratio change) |
| 130 | 0.4538 | S3-sub3 | Continued decline |
| 140 | 0.4489 | S3-sub3 | Early stopping triggered |

### MacroF1 Progression (Base)

| Epoch | Best EMA F1 | Notes |
|-------|-------------|-------|
| 60 | 0.3594 | First best checkpoint |
| 70 | 0.4075 | Rapid improvement |
| 80 | 0.4587 | Surpasses finetune base |
| 90 | **0.4630** | **Final best** |
| 120 | -- | Early stopping triggered |

### Final Training State

**ConvNeXt-Small (E140, stopped):**
```
train_loss=1.556  val_loss=1.309 (cls=0.239, reg=1.070)
macroF1=0.4489  crucial_small_F1=0.2628
```

**ConvNeXt-Base (E120, stopped):**
```
train_loss=1.556  val_loss=1.301 (cls=0.239, reg=1.070)
macroF1 (at stop)  crucial_small_F1=0.2791
```

---

## Full COCO Evaluation Results

### ConvNeXt-Small Phase E (best checkpoint, E110)

```
mAP @ IoU=0.50           = 0.7388
mAP @ IoU=0.75           = 0.5053
mAP @ IoU=0.50:0.95      = 0.4810
mIoU (TP detections)     = 0.7895
```

Per-class breakdown:

| Class | N_GT | AP@50 | AP@75 | AP@50:95 | Prec | Rec | F1 |
|-------|------|-------|-------|----------|------|-----|-----|
| bicycle | 6088 | 0.670 | 0.378 | 0.391 | 0.319 | 0.761 | 0.450 |
| bus | 3982 | 0.818 | 0.658 | 0.588 | 0.409 | 0.864 | 0.555 |
| car | 61405 | 0.833 | 0.633 | 0.579 | 0.551 | 0.856 | 0.671 |
| motorcycle | 3642 | 0.696 | 0.406 | 0.398 | 0.331 | 0.778 | 0.464 |
| person | 59584 | 0.721 | 0.471 | 0.449 | 0.464 | 0.751 | 0.574 |
| scooter | 3970 | 0.460 | 0.252 | 0.258 | 0.321 | 0.516 | 0.395 |
| truck | 8828 | 0.739 | 0.528 | 0.493 | 0.396 | 0.791 | 0.528 |
| bollard | 50129 | 0.540 | 0.309 | 0.313 | 0.456 | 0.572 | 0.508 |
| traffic_light_red* | 5497 | 0.743 | 0.337 | 0.380 | 0.116 | 0.821 | 0.203 |
| neubie* | 980 | 0.867 | 0.721 | 0.659 | 0.179 | 0.889 | 0.298 |
| warning_light* | 2088 | 0.838 | 0.675 | 0.605 | 0.235 | 0.852 | 0.368 |
| ev_close | 1231 | 0.928 | 0.806 | 0.732 | 0.490 | 0.942 | 0.644 |
| ev_open* | 193 | 0.826 | 0.687 | 0.603 | 0.050 | 0.938 | 0.096 |
| cart | 1151 | 0.835 | 0.598 | 0.564 | 0.397 | 0.861 | 0.543 |
| traffic_light_green* | 2038 | 0.750 | 0.361 | 0.395 | 0.112 | 0.834 | 0.197 |
| traffic_light_other* | 12071 | 0.558 | 0.268 | 0.290 | 0.073 | 0.704 | 0.132 |

`*` = rare class, lower score threshold (0.08) applied at P/R/F1 computation.

### ConvNeXt-Base Phase E (best checkpoint, E90)

```
mAP @ IoU=0.50           = 0.7302
mAP @ IoU=0.75           = 0.4947
mAP @ IoU=0.50:0.95      = 0.4714
mIoU (TP detections)     = 0.7880
```

Per-class breakdown:

| Class | N_GT | AP@50 | AP@75 | AP@50:95 | Prec | Rec | F1 |
|-------|------|-------|-------|----------|------|-----|-----|
| bicycle | 6088 | 0.668 | 0.375 | 0.390 | 0.347 | 0.750 | 0.475 |
| bus | 3982 | 0.806 | 0.653 | 0.583 | 0.381 | 0.864 | 0.528 |
| car | 61405 | 0.827 | 0.624 | 0.571 | 0.610 | 0.841 | 0.707 |
| motorcycle | 3642 | 0.689 | 0.396 | 0.393 | 0.328 | 0.769 | 0.460 |
| person | 59584 | 0.718 | 0.471 | 0.448 | 0.464 | 0.750 | 0.573 |
| scooter | 3970 | 0.438 | 0.215 | 0.236 | 0.316 | 0.505 | 0.389 |
| truck | 8828 | 0.722 | 0.522 | 0.484 | 0.389 | 0.785 | 0.520 |
| bollard | 50129 | 0.531 | 0.302 | 0.306 | 0.432 | 0.568 | 0.491 |
| traffic_light_red* | 5497 | 0.745 | 0.333 | 0.378 | 0.119 | 0.824 | 0.207 |
| neubie* | 980 | 0.873 | 0.720 | 0.652 | 0.191 | 0.896 | 0.314 |
| warning_light* | 2088 | 0.841 | 0.666 | 0.597 | 0.283 | 0.853 | 0.425 |
| ev_close | 1231 | 0.912 | 0.793 | 0.716 | 0.498 | 0.926 | 0.648 |
| ev_open* | 193 | 0.797 | 0.634 | 0.561 | 0.051 | 0.959 | 0.096 |
| cart | 1151 | 0.822 | 0.590 | 0.556 | 0.400 | 0.865 | 0.548 |
| traffic_light_green* | 2038 | 0.748 | 0.364 | 0.390 | 0.139 | 0.833 | 0.239 |
| traffic_light_other* | 12071 | 0.546 | 0.259 | 0.282 | 0.075 | 0.695 | 0.136 |

### Comparison With Previous Best (ViT-S+ mix, E188)

| Metric | ViT-S+ mix (E188) | Phase E Small (E110) | Phase E Base (E90) |
|--------|-------------------|---------------------|-------------------|
| mAP@50 | 0.5289 | **0.7388** (+39.7%) | **0.7302** (+38.1%) |
| mAP@50:95 | 0.3346 | **0.4810** (+43.8%) | **0.4714** (+40.9%) |
| MacroF1 (EMA) | 0.6919 | 0.4576 | 0.4630 |

Note: mAP metrics are substantially better with Phase E. The F1 numbers are not directly comparable because the ViT-S+ mix eval used the buggy centerness-halved scores with thresholds tuned to that scale, while Phase E eval uses corrected AABB decode.

---

## Critical Bug Fixes (Sep 5-6)

Three bugs present in ALL checkpoints since v2:

### 1. Centerness Score Halving (CRITICAL)

When `USE_CENTERNESS=False`, head outputs `ctr = zeros(...)`. Decode computes:
```
final_score = sigmoid(0) * cls_score = 0.5 * cls_score
```
Every detection score was silently halved. All calibrated thresholds were tuned against half-strength scores.

**Verified:** Score ratio buggy/correct = exactly 0.5000 on all test images.

**Fix:** `decode_outputs_aabb()` in `src/utils_OBB.py` -- does not use centerness.

### 2. ProBIoU Rotated NMS on AABB Detections (13x speedup)

Using oriented-bounding-box NMS (pure Python loop) for axis-aligned boxes.

**Fix:** `torchvision.ops.batched_nms` (CUDA). Wall-clock: 0.452s to 0.035s.

### 3. DFL Double-Decode Risk (latent)

`decode_outputs_OBB()` re-applies DFL softmax when `reg_ch > 4`, but head already decodes DFL.

**Fix:** Assert `reg_ch == 4` in `decode_outputs_aabb()`.

---

## TRT-Clean Head & Fine-Tuning (Sep 28-29)

### Problem

Phase E head uses **GroupNorm** and **DCNv2** which block efficient TensorRT INT8 deployment:
- GroupNorm: no INT8 kernel in TensorRT -- forces fp16 precision islands
- DCNv2 (`torchvision.ops.DeformConv2d`): no ONNX symbolic -- export fails entirely

### Solution: `src/model_head_convnext_trt.py`

Drop-in replacement for `model_head_convnext.py` with two changes:

| Component | Original | TRT v2 |
|-----------|----------|--------|
| Normalization | GroupNorm (20 instances) | **BatchNorm2d** |
| cls_tower positions 4,5 | DCNv2Block (offset + deformable conv) | **DilatedConvBlockBN** (Conv2d 3×3 dilation=2, effective 5×5 RF + BN + ReLU + residual) |

Everything else unchanged: CrossLevelAttention (LayerNorm), ECABlock, DFL, CosineConv2d, AuxDecoderHead, AIFIEncoder.

**Self-test verified:**
- Forward/backward pass works for both Small and Base configs
- Output shapes match: P2(60x80), P3(30x40), P4(15x20), P5(8x10)
- Zero `DeformConv2d` or `GroupNorm` in model
- Params: ~6.48M (Small), ~6.59M (Base)

### TRT V1 Results & Why V2 Was Needed

**V1 approach** (Sep 28): Plain Conv3x3BlockBN at DCN positions, no BN calibration.

**V1 result**: F1 dropped from **0.4576 → 0.3044** after 15-epoch fine-tune (~33% drop).

**Root cause — BN cold-start problem**: After GN→BN conversion, BatchNorm starts with `running_mean=0, running_var=1`. These incorrect statistics cause severe feature map distortion. 15 epochs wasn't enough for BN running stats to converge, especially with momentum=0.1 exponential moving average.

### TRT V2 Improvements (Sep 29)

Three changes to recover accuracy:

| Change | Why |
|--------|-----|
| **DilatedConvBlockBN** (dilation=2) at DCN positions | Partially recovers DCN's adaptive receptive field — effective 5×5 RF vs plain 3×3 |
| **`calibrate_bn_stats()`** — forward-only pass over ~1000 batches | Populates correct BN `running_mean` / `running_var` BEFORE fine-tuning starts. Backbone frozen, head in train mode, no gradients. |
| **`freeze_bn()`** — locks BN stats during fine-tune | After calibration, keeps BN in eval mode so running stats don't drift. Only conv weights learn. |

### Weight Conversion Function

`convert_phaseE_to_trt(src_path, dst_path)` at bottom of file:
- Detects BN layers via 1-D weight heuristic (handles Sequential-indexed layers like `fpn.proj_s8.1.weight`)
- Excludes LayerNorm in CrossLevelAttention and AuxDecoder paths via `_ln_markers` list
- `.gn.weight/.bias` → `.bn.weight/.bias` (same tensor, rename only)
- Adds `running_mean=0`, `running_var=1`, `num_batches_tracked=0` for all BN layers
- `.dcn.weight` → `.conv.weight` (same shape [256,256,3,3], direct copy)
- `.dcn.bias` → `.conv.bias` (direct copy)
- Drops all `.offset_mask.*` keys (no longer needed)

### Training Script Modifications

Added to `train/train_convnext.py`:

| Flag | Purpose |
|------|---------|
| `--trt-head` | Switches import to `src/model_head_convnext_trt.py` |
| `--calibrate-bn N` | Runs N-batch forward-only BN calibration before training starts. Saves `bn_calibrated.pth` checkpoint. |
| `--freeze-bn` | Freezes BN running stats during fine-tuning (re-applied after every `model.train()` call) |

### Inference Auto-Detection

`inference/inference_batch_v2.py` auto-detects TRT checkpoints by probing for `.bn.weight` keys in the state dict — no `--trt-head` flag needed at inference time.

### Fine-Tuning Commands (V2)

```bash
# Step 1: Convert Phase E checkpoints (v2 conversion with dilated conv support)
python -c "from src.model_head_convnext_trt import convert_phaseE_to_trt; convert_phaseE_to_trt('results/mixed-dataset-training/cnx_small_phaseE/best.pth', 'results/mixed-dataset-training/cnx_small_phaseE/best_trt_v2.pth')"
python -c "from src.model_head_convnext_trt import convert_phaseE_to_trt; convert_phaseE_to_trt('results/mixed-dataset-training/cnx_base_phaseE/best.pth', 'results/mixed-dataset-training/cnx_base_phaseE/best_trt_v2.pth')"

# Step 2: Fine-tune Small (GPU 0, with BN calibration + frozen BN)
nohup python -u train/train_convnext.py \
    --config config/config_convnext_small.py \
    --resume results/mixed-dataset-training/cnx_small_phaseE/best_trt_v2.pth \
    --finetune --finetune-epochs 15 --finetune-lr 5e-5 \
    --trt-head --calibrate-bn 1000 --freeze-bn \
    --batch-size 56 --device cuda:0 \
    --use-amp --run-name cnx_small_trt_v2_finetune \
    > nohup_trt_finetune_small.out 2>&1 &

# Step 3: Fine-tune Base (GPU 2, with BN calibration + frozen BN)
nohup python -u train/train_convnext.py \
    --config config/config_convnext_base.py \
    --resume results/mixed-dataset-training/cnx_base_phaseE/best_trt_v2.pth \
    --finetune --finetune-epochs 15 --finetune-lr 5e-5 \
    --trt-head --calibrate-bn 1000 --freeze-bn \
    --batch-size 56 --device cuda:2 \
    --use-amp --run-name cnx_base_trt_v2_finetune \
    > nohup_trt_finetune_base.out 2>&1 &
```

**Fine-tune LR = 5e-5** (half normal finetune LR — BN is frozen so only conv weights learn).

**Finetune mode behavior** (`--finetune`):
- Freezes COCO cls head
- Uses single LR for shared + neubie params
- Sets `VFL_Q_FLOOR=0.0`, `AUX_LOSS_WEIGHT=0.0`
- Runs for `--finetune-epochs` from resume point

---

## Deliverable Package (Sep 16)

`deliverable/` directory -- self-contained package for developer handoff:

| File | Purpose |
|------|---------|
| `convnext_head.py` | 700-line single file with all head blocks, ONNX-safe DCN (`DeformConv2dGridSample`), `decode_outputs_aabb()`, `load_for_export()` with auto-detection |
| `convnext_backbone.py` | Backbone wrapper returning 3 multi-scale feature maps |
| `preprocess.py` | Image-to-tensor with ImageNet normalization |
| `run_inference.py` | Complete inference script (image/video/directory) |
| `class_names.txt` | 16 Neubie classes |

`load_for_export()` auto-detects checkpoint architecture from key names (cosine_cls, cross_level_attn, DCN, reg_max) and handles DCN weight conversion automatically.

---

## Historical Performance Comparison

All ConvNeXt runs since August:

| Run | Backbone | Architecture | Epochs | Best Metric | Status |
|-----|----------|-------------|--------|-------------|--------|
| ConvNeXt-Small initial | CNX-S | Original head | 180 | F1=0.435 | Complete |
| ConvNeXt-Small calibration | CNX-S | + calibration pass | 180 | F1=0.431 | Complete |
| Finetune Small | CNX-S | CosineConv2d, no DCN | 195 | F1=0.4394 | Complete |
| Finetune Base | CNX-B | CosineConv2d, no DCN | 195 | F1=0.4520 | Complete |
| VFL Q=0 scratch Small | CNX-S | Clean start | 56 | F1=0.2680 | Baseline |
| VFL Q=0 scratch Base | CNX-B | Clean start | 51 | F1=0.3021 | Baseline |
| **Phase E Small** | CNX-S | Conv2d + cross-attn + DCN + aux | 140 (ES) | **F1=0.4576** | **Complete** |
| **Phase E Base** | CNX-B | Conv2d + cross-attn + DCN + aux | 120 (ES) | **F1=0.4630** | **Complete** |
| TRT v1 finetune Small | CNX-S | BN + Conv3x3 (no calibration) | 15 | F1=0.3044 | Complete (failed — BN cold-start) |
| **TRT v2 finetune Small** | CNX-S | BN + DilatedConv + BN-cal + freeze-BN | 15 | — | **In progress** |
| **TRT v2 finetune Base** | CNX-B | BN + DilatedConv + BN-cal + freeze-BN | 15 | — | **Ready to launch** |

---

## Checkpoints

### Phase E Best Checkpoints (use these for evaluation/deployment)

```
results/mixed-dataset-training/cnx_small_phaseE/best.pth   # E110, F1=0.4576
results/mixed-dataset-training/cnx_base_phaseE/best.pth    # E90, F1=0.4630
```

### Periodic Checkpoints (for rollback)

```
results/mixed-dataset-training/cnx_small_phaseE/model_e{010..140}.pth
results/mixed-dataset-training/cnx_base_phaseE/model_e{010..120}.pth
```

### Pretrained Backbone Weights

```
checkpoints/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth   (192 MB)
checkpoints/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth    (338 MB)
```

---

## Files Created / Modified This Sprint

| File | Action | Description |
|------|--------|-------------|
| `src/model_head_convnext_trt.py` | **Created** | TRT-clean head (BN + DilatedConv), `convert_phaseE_to_trt()`, `calibrate_bn_stats()`, `freeze_bn()` |
| `src/model_head_convnext _single_code.py` | **Created** | Self-contained Phase E head (all building blocks inlined, no cross-imports) |
| `train/train_convnext.py` | **Modified** | Added `--trt-head`, `--calibrate-bn`, `--freeze-bn` flags |
| `src/utils_OBB.py` | **Modified** | Added `decode_outputs_aabb()`, `detection_inference_aabb()` |
| `inference/inference_batch_v2.py` | **Modified** | AABB decode path, Phase E config passthrough, threshold CLI overrides, TRT checkpoint auto-detection |
| `inference/inference_images.py` | **Created** | Single-image inference script |
| `config/config_convnext_small.py` | **Modified** | Phase E flags (COSINE_CLS, CROSS_LEVEL_ATTN, DCN, etc.) |
| `config/config_convnext_base.py` | **Modified** | Phase E flags (matching small) |
| `deliverable/convnext_head.py` | **Created** | Self-contained head for developer handoff |
| `deliverable/convnext_backbone.py` | **Created** | Backbone wrapper |
| `deliverable/preprocess.py` | **Created** | Image preprocessing |
| `deliverable/run_inference.py` | **Created** | Complete inference script |
| `deliverable/class_names.txt` | **Created** | 16 Neubie classes |
| `docs/two_week_progress_report_sep4_18.md` | **Created** | Mid-month progress report |

---

## Known Issues

1. **Rare class F1 still low**: traffic_light_other (0.132), ev_open (0.096), traffic_light_green (0.197). These classes have extreme recall/precision imbalance -- high recall (0.70-0.96) but very low precision. Threshold calibration will help but the fundamental issue is class imbalance.

2. **Threshold re-calibration needed**: All threshold files were calibrated against buggy (halved) scores. Phase E checkpoints need fresh calibration using AABB decode path.

3. **eval_convnext.py centerness fix**: One-line fix needed -- add `use_centerness=USE_CENTERNESS` to decode calls. Not blocking but affects reported P/R/F1 numbers.

4. **TRT v1 fine-tune failed**: Replacing GN→BN + DCN→Conv3x3 without BN calibration caused F1 to drop from 0.4576 to 0.3044. BN running stats never converged in 15 epochs. **TRT v2** addresses this with: (a) `calibrate_bn_stats()` forward pass over 1000 batches before training, (b) `freeze_bn()` to lock calibrated stats during fine-tune, (c) `DilatedConvBlockBN` (dilation=2) for larger receptive field at DCN positions.

---

## Next Steps (Priority Order)

1. ~~Convert Phase E checkpoints~~ ✓ Done (`best_trt_v2.pth` for both Small and Base)
2. ~~Launch TRT v2 fine-tuning (Small)~~ ✓ In progress with `--calibrate-bn 1000 --freeze-bn`
3. **Launch TRT v2 fine-tuning (Base)** — same flags, GPU 2
4. **Evaluate TRT v2 fine-tuned models** — compare mAP/F1 against Phase E baselines, verify BN calibration recovered accuracy
5. **Threshold calibration** on TRT fine-tuned checkpoints using AABB decode
6. **ONNX export** of TRT-clean head for TensorRT INT8 deployment
7. **Update deliverable package** with final TRT checkpoint + calibrated thresholds
8. **ROS2 integration** — deploy to `dinov3_ros_tensorrt`

---

## Commands Reference

### Training

```bash
# Phase E (already complete, for reference):
nohup python -u train/train_convnext.py \
  --config config/config_convnext_small.py \
  --use-amp --device cuda:2 > nohup_cnx_small_phaseE.out 2>&1 &

# TRT fine-tune:
nohup python -u train/train_convnext.py \
  --config config/config_convnext_small.py \
  --resume results/mixed-dataset-training/cnx_small_phaseE/best_trt.pth \
  --finetune --finetune-epochs 15 --finetune-lr 5e-5 \
  --trt-head --device cuda:0 --gpus 0,1 \
  --use-amp --run-name cnx_small_trt_finetune \
  > nohup_trt_finetune_small.out 2>&1 &
```

### Evaluation

```bash
python eval_convnext.py \
  --config config/config_convnext_small.py \
  --checkpoint results/mixed-dataset-training/cnx_small_phaseE/best.pth \
  --device cuda:0 --split val
```

### Weight Conversion

```bash
python -c "
from src.model_head_convnext_trt import convert_phaseE_to_trt
convert_phaseE_to_trt('path/to/best.pth', 'path/to/best_trt.pth')
"
```

### Batch Inference

```bash
python inference/inference_batch_v2.py \
  --config config/config_convnext_small.py \
  --device cuda:0 --tile 3 --aabb --show-fps \
  --score-thresh 0.35 --rare-thresh 0.20 --nms-thresh 0.45
```

### Self-Test TRT Head

```bash
python src/model_head_convnext_trt.py                      # basic self-test
python src/model_head_convnext_trt.py path/to/best.pth     # + weight conversion test
```

---

## Files Reference

```
# Configs
config/config_convnext_small.py
config/config_convnext_base.py

# Training
train/train_convnext.py                          # --trt-head flag added

# Head variants
src/model_head_convnext.py                       # Original (GN + DCN)
src/model_head_convnext_trt.py                   # TRT-clean v2 (BN + DilatedConv + calibrate/freeze BN)
src/model_head_convnext _single_code.py          # Phase E self-contained (all blocks inlined)

# Backbone, loss, utils, dataset (shared)
src/model_backbone_convnext.py
src/loss_v3.py
src/utils_OBB.py                                 # includes decode_outputs_aabb()
src/dataset_coco_v3.py
src/class_names.txt

# Evaluation & inference
eval_convnext.py
inference/inference_batch_v2.py
calibrate_thresholds.py

# Deliverable
deliverable/convnext_head.py
deliverable/convnext_backbone.py
deliverable/preprocess.py
deliverable/run_inference.py
deliverable/class_names.txt

# Checkpoints
results/mixed-dataset-training/cnx_small_phaseE/
results/mixed-dataset-training/cnx_base_phaseE/
results/mixed-dataset-training/cnx_small_phaseE/best_trt_v2.pth  # converted
results/mixed-dataset-training/cnx_base_phaseE/best_trt_v2.pth   # converted
results/mixed-dataset-training/cnx_small_trt_v2_finetune/  # in progress
results/mixed-dataset-training/cnx_base_trt_v2_finetune/   # ready to launch
checkpoints/dinov3_convnext_small_pretrain_lvd1689m-*.pth
checkpoints/dinov3_convnext_base_pretrain_lvd1689m-*.pth

# Training logs
nohup_cnx_small_phaseE.out
nohup_cnx_base_phaseE.out
eval_phaseE_small.out
eval_phaseE_base.out
```
