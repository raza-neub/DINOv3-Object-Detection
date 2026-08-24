# Experiment 3: v3 Multi-Depth FPN + VariFocal Loss (ViT-S/16+)

## Status: COMPLETED, ARCHIVED (1 training run)

## Summary

Key architectural upgrade: instead of deriving all FPN levels from a single (final) backbone block, tap the backbone at 3 depths (shallow/mid/deep) to seed each FPN level with features at the appropriate abstraction level. Also replaced focal loss with VariFocal Loss (VFL) and added mosaic augmentation.

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Config | (CLI overrides on config.py) | run_config.json captures actual settings |
| Backbone | `src/model_backbone_v3.py` | `DinoBackboneV3`: 3 outputs from blocks [n//4, n//2, n-1] |
| Head | `src/model_head_v3.py` | MultiLevelEfficientPAN + DetectHeadV3 (per-level prediction convolutions) |
| Loss | `src/loss_v3.py` | VariFocal Loss (VFL), optional DFL, optional centerness |
| Utils | `src/utils_OBB.py` | OBB decode (re-exports AABB decode for non-OBB mode) |
| Dataset | `src/dataset_coco_v3.py` | Mosaic, flip, translate, photometric augmentation |
| Optimizer | `src/optimizer.py` | MuSGD (Muon+SGD hybrid) |

## Architecture

```
Input (480x640) --> DinoBackboneV3 (ViT-S/16+, frozen, 384-d)
  --> 3 feature maps at same resolution (30x40) from different depths:
      shallow (block 3):  local textures/edges  --> seeds P3 (small objects)
      mid     (block 6):  part-level features   --> seeds P4
      deep    (block 11): global semantics       --> seeds P5 (large objects)
  --> MultiLevelEfficientPAN (bidirectional, 3-level, 192ch)
      - Projects shallow/mid/deep to 192ch
      - Top-down (coarse->fine) + bottom-up (fine->coarse) with RepDWSBlock
      - SmallObjectRefine on P3
  --> DetectHeadV3 (per-level prediction convolutions)
      - Shared towers: cls_tower, reg_tower (RepDWSBlock x N_CONVS)
      - Per-level: cls_logits[l], bbox_reg[l], centerness[l], angle_reg[l]
      - Each FPN level gets its OWN 3x3 final prediction layer
  --> cls (VFL), reg (CIoU), ctr (BCE), angle (sin^2)
```

## Key Improvements Over v2

1. **Multi-depth backbone**: v2 derives all FPN levels from `feats[-1]` (only global semantics). v3 taps shallow/mid/deep blocks so P3 gets local texture features ideal for small objects.
2. **Per-level prediction heads**: v2 shares cls_logits/bbox_reg across all levels. v3 gives each FPN level its own final prediction convolution with independent biases, allowing level specialization.
3. **VariFocal Loss**: Replaces focal loss. IoU-aware quality scores as soft targets, with class weights applied only to positives (preserves rare-class recall without suppressing background).
4. **Mosaic augmentation**: 4-image random tiling (50% probability), with late-epoch closure.
5. **Geometric augmentation**: Horizontal flip (50%), random translate (50%, max 10%).
6. **EMA**: Exponential Moving Average with tau warmup (decay=0.9999, tau=2000).

## Backbone Detail (`src/model_backbone_v3.py`)

```python
class DinoBackboneV3(nn.Module):
    def __init__(self, dino_model, n_layers):
        # For 12-layer ViT: taps at layers [3, 6, 11]
        self.layer_indices = [n_layers // 4, n_layers // 2, n_layers - 1]

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=self.layer_indices, reshape=True)
        return list(feats)  # [shallow, mid, deep], each (B, 384, 30, 40)
```

## Loss Details (`src/loss_v3.py`)

**VariFocal Loss (VFL)**:
```
q = target_score (from STAL, in [0, 1])
Positives (q > 0): loss = -q * [q*log(sigma) + (1-q)*log(1-sigma)]
Negatives (q = 0): loss = -alpha * sigma^gamma * log(1-sigma)
```
- Class weights applied ONLY to positives (increase rare-class recall without suppressing background)
- Per-class bias initialized to focal-style prior

**Other loss components**: Same STAL assignment, CIoU reg, BCE centerness, sin^2(2*dtheta) angle, ProgLoss warmup as v2.

## Training Run

### Run: 2026-05-13 (120 epochs, early-stopped from 200)

```bash
nohup python -u train/train_v3.py --config config/config_v3.py \
  --device cuda:1 --batch-size 64 --epochs 200 --lr 3e-4 --patience 20 --use-amp \
  > nohup_v3.out 2>&1 &
```

**Config settings**:
```python
MULTI_LAYER_BACKBONE = True
USE_MOSAIC           = True
MOSAIC_PROB          = 0.5
CLOSE_MOSAIC_EPOCHS  = 10
USE_VFL              = True
USE_EMA              = True
EMA_DECAY            = 0.9999
EMA_TAU              = 2000
HFLIP_PROB           = 0.5
TRANSLATE_PROB       = 0.5
TRANSLATE_MAX        = 0.1
```

**Results** (epoch 119, early-stopped):
```
val_loss: 0.7275 (cls=0.1340, reg=0.3922, ctr=0.1967, angle=0.0227)
mosaic_on=true, ema_decay=0.9999
```

**Checkpoint**: `results/2026-05-13_07-34-29_dinov3_vits16plus_v3/` (120 checkpoints, best at epoch 99)

## Why It Was Dropped

1. **Single-dataset only**: Trained on Neubie data only. No mechanism to leverage COCO's 80-class population for transfer learning. COCO's huge person/vehicle/traffic-light data improves backbone feature discrimination.
2. **No dual-classifier**: Cannot jointly train on COCO + Neubie without class-set conflicts. The mix pipeline solves this with separate `cls_coco` (80) and `cls_neubie` (16) heads.
3. **Higher val_loss than v2**: 0.7275 vs 0.6474 (v2). The VFL + mosaic combination needed more epochs and the curriculum training of the mix pipeline to converge properly.
4. **Missing class-imbalance handling**: No RepeatFactorSampler, no per-class weights, no per-class thresholds. Rare classes (scooter, bollard, traffic_light_other) get drowned.

## What Was Kept for Later Versions

- `src/model_backbone_v3.py`: Multi-depth backbone reused in all subsequent pipelines (mix, mix_p2)
- `src/model_head_v3.py`: MultiLevelEfficientPAN + DetectHeadV3 is the base for mix/mix_p2 heads
- `src/loss_v3.py`: VFL + STAL + optional DFL is the loss for all subsequent pipelines
- `src/dataset_coco_v3.py`: Mosaic + geometric augmentation pipeline reused everywhere
- The per-level prediction head idea (independent biases per FPN level) is kept in all mix variants
