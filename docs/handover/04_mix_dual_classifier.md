# Experiment 4: Mix Dual-Classifier (COCO + Neubie, ViT-S/16+)

## Status: COMPLETED — Best ViT-S+ baseline (F1=0.6919, mAP@50=0.5289)

## Summary

The breakthrough experiment. Dual-classifier head enables joint COCO (80 classes) + Neubie (16 classes) training via a 4-stage curriculum. COCO's large-scale data transfers general detection capability; Neubie-specific classifier fine-tunes to the deployment domain. Added CosineConv2d classifiers for rare-class calibration, decoupled cls/reg towers, RepeatFactorSampler for class imbalance.

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config | `config/config_mix.py` | Standalone config: 250 epochs, 4-stage curriculum, BS=128 |
| Backbone | `src/model_backbone_v3.py` | `DinoBackboneV3`: 3 depths [3, 6, 11] from ViT-S/16+ |
| Head | `src/model_head_mix.py` | `DINODetectionHeadMix`: dual classifiers (cls_coco + cls_neubie) |
| Loss | `src/loss_v3.py` | VFL + STAL + CIoU + BCE centerness + ProgLoss |
| Utils | `src/utils_OBB.py` | OBB decode (axis-aligned mode), traffic-light cross-class NMS |
| Dataset | `src/dataset_coco_v3.py` | `DatasetCOCOv3`: mosaic, flip, translate, photometric |
| Training | `train/train_mix_p2.py` | 4-stage curriculum trainer with per-component LRs |
| Eval | `eval_mix.py` | Per-class AP@50, AP@50:95, P/R/F1 |
| Inference | `inference/inference_batch_v2.py` | Auto-detects mix head, video batch inference |

## Architecture

```
Input (480x640) --> DinoBackboneV3 (ViT-S/16+, frozen, 384-d)
  --> [shallow, mid, deep] each (B, 384, 30, 40)
  --> MultiLevelEfficientPAN (bidirectional, 3-level P3/P4/P5, 192ch)
  --> Decoupled towers:
      cls_entry: 1x1 expand 192ch -> 256ch
      cls_tower: 6 RepDWSBlock @ 256ch (wider, deeper)
      reg_tower: 4 RepDWSBlock @ 192ch (lightweight)
  --> ECA channel attention at cls output
  --> Dual classifiers (per-level):
      cls_coco[L]:   CosineConv2d, 80 classes (COCO auxiliary)
      cls_neubie[L]: CosineConv2d, 16 classes (Neubie, deployed)
  --> reg (CIoU, LTRB softplus), ctr (BCE)
  --> forward(feats, dataset='coco'|'neubie') selects classifier
```

## Key Innovations Over v3

1. **Dual classifiers**: `cls_coco` (80 classes) and `cls_neubie` (16 classes) share the same FPN + towers but have independent final layers. `forward(feats, dataset='coco')` or `forward(feats, dataset='neubie')` selects which classifier to use.

2. **CosineConv2d** (long-tail fix): `scale * <f_hat, w_hat> + bias` where f_hat, w_hat are L2-normalized. Decouples classification from magnitude — rare-class weight norms don't get suppressed by majority classes. Learnable temperature (log-space) and per-class bias.

3. **Decoupled cls/reg towers**: Classification gets more capacity (6 RepDWS @ 256ch) than regression (4 @ 192ch). Classification is harder (16 fine-grained classes); regression is comparatively easy.

4. **4-stage curriculum training** (see below).

5. **RepeatFactorSampler**: Oversamples rare Neubie classes (t=0.005, max_factor=4.0).

6. **Per-component learning rates**: Separate LR groups for shared, cls_coco, cls_neubie.

## 4-Stage Curriculum

```
Stage 1 (epochs 0-49):   COCO ONLY
  - Warm shared head + COCO classifier
  - LR_SHARED=2e-4, LR_CLS_COCO=1e-3, LR_CLS_NEUBIE=0
  - Purpose: build general detection features from COCO's 80-class diversity

Stage 2 (epochs 50-69):  NEUBIE ONLY
  - Warm Neubie classifier, low LR on shared (preserve COCO features)
  - LR_SHARED=2e-5, LR_CLS_COCO=0, LR_CLS_NEUBIE=1e-3
  - Purpose: bootstrap Neubie classifier while shared features are frozen

Stage 3 (epochs 70-229): MIXED COCO + NEUBIE
  - Shifting ratio: 60/40 -> 40/60 -> 20/80 (COCO/Neubie)
  - Sub-stages: ep 70-109 (60/40), ep 110-144 (40/60), ep 145-229 (20/80)
  - LR_SHARED=2e-4, both classifiers active
  - MixedBatchIterator interleaves COCO/Neubie batches to enforce ratio
  - Purpose: joint optimization, gradually specializing to Neubie

Stage 4 (epochs 230-249): NEUBIE ONLY (calibration)
  - LR_SHARED=2e-4, both cls active
  - Purpose: final Neubie-only calibration pass
```

## Full Config (`config/config_mix.py`)

```python
# Backbone
DINO_MODEL   = 'dinov3_vits16plus'    # 384-d embed, 12 layers
IMG_SIZE     = (480, 640)             # H, W

# Head
FPN_CH       = 192
N_CONVS      = 4
NUM_COCO_CLASSES   = 80
NUM_NEUBIE_CLASSES = 16

# Augmentation
MOSAIC_PROB  = 0.5
HUE_DELTA    = 10                     # original value (predates hue-fix)
HFLIP_PROB   = 0.5
TRANSLATE_PROB = 0.5

# Training
BATCH_SIZE   = 128
NUM_EPOCHS   = 250
LR_SHARED    = 2e-4
LR_CLS_COCO  = 1e-3
LR_CLS_NEUBIE = 1e-3
WEIGHT_DECAY = 0.0001

# Loss
USE_VFL      = True
VFL_ALPHA    = 0.75
USE_OBB      = False                  # axis-aligned (no angle branch)
TAL_TOPK     = 12
PROG_LOSS_EPOCHS = 10

# EMA
EMA_DECAY    = 0.999
EMA_TAU      = 2000

# Validation
BEST_METRIC  = 'f1'                   # macro-F1 for checkpoint selection
VAL_METRIC_EVERY = 3
```

## Training Runs (6 iterations to reach best)

### Run 1: 2026-05-14 (stopped early at epoch 29)
```bash
nohup python -u train/train_mix_p2.py --config config/config_mix.py \
  --batch-size 64 --use-amp > nohup_mix_run1.out 2>&1 &
```
- Initial exploration, LR_SHARED=1e-4, LR_CLS=5e-4
- Stopped early — hyperparameters not tuned

### Run 2: 2026-05-15 (full 200 epochs)
- Updated LR_SHARED=2e-4, LR_CLS=1e-3
- Reached epoch 199, train_loss=0.722, val_loss=0.718
- First successful end-to-end mix training

### Run 3: 2026-05-21 (169 epochs)
- Added persistent workers, memory pinning, periodic saving
- train_loss=0.874, val_loss=0.801

### Run 4: 2026-05-28 (resumed from run 3, extended to 450 epochs)
- Extended Stage 3 to epoch 390
- Reached epoch 199, train_loss=0.792, val_loss=0.787

### Run 5: 2026-06-01 (resumed, 250 epochs)
- Shortened schedule, reached epoch 200
- train_loss=0.886, val_loss=0.805

### Run 6: 2026-06-01 — BEST MODEL (`mix_best_20260604`)
```bash
nohup python -u train/train_mix_p2.py --config config/config_mix.py \
  --batch-size 128 --num-workers 16 --use-amp > nohup_mix_best.out 2>&1 &
```

**Final Config**: BATCH_SIZE=128, EMA_DECAY=0.999, VAL_METRIC_EVERY=3, 16 workers, pin_memory

**Results** (reached epoch 229):
```
train_loss: 0.874
val_loss_neubie: 0.794 (cls=0.205, reg=0.387, ctr=0.196, ang=0.023)
```

**Checkpoint**: `results/mixed-dataset-training/mix_best_20260604/`
- `best_e182.pth` — F1=0.6868
- `best_e188.pth` — F1=0.6859 (best SAHI TL recall)
- `best_e250.pth` — **F1=0.6919** (final best)

## Formal Evaluation Results

### eval_e182.out (best_e182.pth)
```
mAP @ IoU=0.50:      0.5281
mAP @ IoU=0.50:0.95: 0.3336
mIoU (TP):            0.7755
Inference speed:      3.7 img/s (33,276 images in 9049s)

Per-class AP@50:
  ev_close:            0.8430    (strong)
  neubie:              0.7098    (strong)
  warning_light:       0.6176    (moderate)
  bus:                 0.6088
  car:                 0.6071
  ev_open:             0.5769
  person:              0.5665
  truck:               0.5656
  motorcycle:          0.5524
  bicycle:             0.5303
  traffic_light_red:   0.4825
  cart:                0.4507
  traffic_light_green: 0.4227
  bollard:             0.2965    (weak)
  traffic_light_other: 0.2834    (weak)
  scooter:             0.2317    (weak)
```

### eval_e188.out (best_e188.pth)
```
mAP @ IoU=0.50:      0.5289 (marginal improvement)
mAP @ IoU=0.50:0.95: 0.3346
mIoU (TP):            0.7756
```
Nearly identical — confirms convergence plateau.

## Evaluation Command

```bash
python eval_mix.py --config config/config_mix.py \
  --checkpoint results/mixed-dataset-training/mix_best_20260604/best_e182.pth \
  --device cuda:0 --split val
```

## Calibration

```bash
python calibrate_thresholds.py --config config/config_mix.py \
  --checkpoint results/mixed-dataset-training/mix_best_20260604/best_e250.pth \
  --model-type mix --device cuda:0
```

## Batch Inference

```bash
python inference/inference_batch_v2.py --config config/config_mix.py \
  --device cuda:0 --tile 3 --aabb --show-fps
```

## Why We Moved On (But Did NOT Drop mix)

The mix baseline is preserved as the reference checkpoint. But it plateaued:

1. **mAP@50 stuck at 0.53**: Despite 250 epochs and 6 training iterations, mAP@50 could not break 0.53.
2. **Small-object recall poor**: scooter (0.23), bollard (0.30), traffic_light_other (0.28). All stride-16 features — no stride-8 resolution for tiny objects.
3. **Inference speed**: 3.7 img/s is far from edge deployment target (25-40 FPS).
4. **Box quality gap**: mAP@50 >> mAP@50:95 (0.53 vs 0.33) indicates coarse localization.

The `best_e250.pth` checkpoint became the warm-start for mix_p2 (next experiment).

## Files Reference

```
config/config_mix.py                          # Config
train/train_mix_p2.py                         # Training script (shared with mix_p2)
eval_mix.py                                   # Evaluation
src/model_backbone_v3.py                      # Backbone
src/model_head_mix.py                         # Detection head
src/loss_v3.py                                # Loss
src/utils_OBB.py                              # Decode + NMS
src/dataset_coco_v3.py                        # Dataset loader
src/class_names.txt                           # 16 Neubie classes
inference/inference_batch_v2.py               # Batch video inference
calibrate_thresholds.py                       # Per-class threshold tuning
results/mixed-dataset-training/mix_best_20260604/  # Checkpoints
```
