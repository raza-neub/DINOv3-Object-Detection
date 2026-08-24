# DINOv3 Object Detection

Anchor-free FCOS-style object detection heads trained on top of **frozen DINOv3** backbones (ViT-S/16+ and ConvNeXt). Designed for deployment via ONNX/TensorRT on Jetson Orin.

## Pipelines

| Pipeline | Backbone | FPN Levels | Key Feature |
|---|---|---|---|
| `vitsplus_mix` | ViT-S/16+ | P3/P4/P5 (3-level) | Dual classifier (COCO+Neubie), cosine head |
| `vitsplus_mix_p2` | ViT-S/16+ | P2/P3/P4/P5 (4-level) | Stride-8 P2 for small objects + DFL/DCN/AIFI |
| `convnext_small` | ConvNeXt-Small | P2/P3/P4/P5 (4-level) | Native multi-scale backbone |
| `convnext_base` | ConvNeXt-Base | P2/P3/P4/P5 (4-level) | Larger ConvNeXt model |

## Training

Each pipeline has a dedicated training script. Configs are loaded by file path via `--config`.

```bash
# ViT-S/16+ mix (3-level)
nohup python -u train/train_vitsplus_mix.py \
  --config config/config_vitsplus_mix.py --use-amp > nohup_mix.out 2>&1 &

# ViT-S/16+ mix P2 (4-level, small-object)
nohup python -u train/train_vitsplus_mix_p2.py \
  --config config/config_vitsplus_mix_p2.py --use-amp > nohup_mix_p2.out 2>&1 &

# ConvNeXt-Small
nohup python -u train/train_convnext_small.py \
  --config config/config_convnext_small.py --use-amp > nohup_convnext_small.out 2>&1 &

# ConvNeXt-Base
nohup python -u train/train_convnext_base.py \
  --config config/config_convnext_base.py --use-amp > nohup_convnext_base.out 2>&1 &
```

**Common training flags:**

| Flag | Description |
|---|---|
| `--config PATH` | Config file (required) |
| `--use-amp` | Mixed-precision training |
| `--batch-size N` | Override config batch size |
| `--epochs N` | Override config epoch count |
| `--device cuda:0` | GPU device |
| `--resume PATH` | Resume from checkpoint |
| `--warm-start PATH` | Load head weights only (new training schedule) |
| `--run-name NAME` | Custom name appended to results directory |
| `--patience N` | Early stopping patience (default: 30) |
| `--save-every-epochs N` | Periodic checkpoint interval (default: 10) |
| `--unfreeze-last-n N` | Unfreeze last N backbone layers (default: 0 = frozen) |
| `--num-workers N` | Dataloader workers (default: 8) |

ConvNeXt scripts also support `--compile-backbone` (torch.compile) and `--calibrate` / `--calibrate-epochs` / `--calibrate-lr` for BN calibration.

## Evaluation

COCO-style mAP evaluation with per-class breakdown.

```bash
# ViT-S/16+ mix
python eval/eval_vitsplus_mix.py \
  --config config/config_vitsplus_mix.py --split val

# ViT-S/16+ mix P2
python eval/eval_vitsplus_mix_p2.py \
  --config config/config_vitsplus_mix_p2.py --split val

# ConvNeXt-Small
python eval/eval_convnext_small.py \
  --config config/config_convnext_small.py --split val

# ConvNeXt-Base
python eval/eval_convnext_base.py \
  --config config/config_convnext_base.py --split val
```

**Common eval flags:**

| Flag | Description |
|---|---|
| `--config PATH` | Config file (required) |
| `--checkpoint PATH` | Checkpoint to evaluate (defaults to config's `MODEL_PATH_INFERENCE`) |
| `--split val\|train` | Dataset split (default: val) |
| `--batch-size N` | Eval batch size (default: 8) |
| `--score-thresh F` | Override config score threshold |
| `--nms-thresh F` | Override config NMS threshold |
| `--iou-thresh F` | IoU threshold for AP calculation (default: 0.5) |
| `--max-det N` | Max detections per image (default: 300) |
| `--max-images N` | Limit number of eval images (-1 = all) |
| `--fuse` | Fuse Conv+BN for faster inference |
| `--device cuda:0` | GPU device |

## Inference

Batch video inference with tiled output and optional SAHI (Slicing Aided Hyper Inference).

```bash
# ViT-S/16+ mix P2 — single video
python inference/inference_vitsplus_mix_p2.py \
  --config config/config_vitsplus_mix_p2.py \
  --videos /path/to/video.mp4

# ConvNeXt-Small — multiple videos with SAHI
python inference/inference_convnext_small.py \
  --config config/config_convnext_small.py \
  --videos /path/to/vid1.mp4 /path/to/vid2.mp4 \
  --sahi --sahi-cols 2 --sahi-overlap 0.2

# ConvNeXt-Base — custom output directory
python inference/inference_convnext_base.py \
  --config config/config_convnext_base.py \
  --videos /path/to/video.mp4 --out-dir results/inference/
```

**Common inference flags:**

| Flag | Description |
|---|---|
| `--config PATH` | Config file (required) |
| `--videos PATH [PATH ...]` | Input video files |
| `--checkpoint PATH` | Checkpoint (defaults to config's `MODEL_PATH_INFERENCE`) |
| `--out-dir PATH` | Output directory |
| `--tile N` | Grid tile layout, e.g. 3 = 3x3 (default: 3) |
| `--tile-w N` / `--tile-h N` | Tile dimensions in pixels (default: 640x360) |
| `--sahi` | Enable SAHI (sliced inference for small objects) |
| `--sahi-cols N` | SAHI horizontal slices (default: 2) |
| `--sahi-overlap F` | SAHI slice overlap fraction (default: 0.2) |
| `--score-thresh F` | Override score threshold |
| `--nms-thresh F` | Override NMS threshold |
| `--thresholds-json PATH` | Per-class thresholds from calibration |
| `--aabb` | Use axis-aligned boxes instead of OBB |
| `--fuse` | Fuse Conv+BN for faster inference |
| `--show-fps` | Display FPS counter |
| `--device cuda:0` | GPU device |

## Threshold Calibration

Per-class score threshold tuning to maximize F1 on the validation set. Outputs a JSON file that can be passed to inference via `--thresholds-json`.

```bash
# Calibrate ViT-S/16+ mix P2
python calibrate_thresholds.py \
  --config config/config_vitsplus_mix_p2.py \
  --checkpoint results/<run>/best.pth \
  --model-type mix_p2 \
  --output thresholds_mix_p2.json

# Calibrate ConvNeXt-Small
python calibrate_thresholds.py \
  --config config/config_convnext_small.py \
  --checkpoint results/<run>/best.pth \
  --model-type convnext \
  --output thresholds_convnext_small.json
```

**Calibration flags:**

| Flag | Description |
|---|---|
| `--config PATH` | Config file (required) |
| `--checkpoint PATH` | Checkpoint to calibrate (required) |
| `--model-type mix\|mix_p2\|convnext\|v3` | Model architecture type (default: mix) |
| `--output PATH` | Output JSON path (default: auto-generated) |
| `--batch-size N` | Batch size (default: 16) |
| `--base-thresh F` | Minimum threshold to search from (default: 0.01) |
| `--thresh-steps N` | Number of threshold steps to search (default: 200) |
| `--iou-thresh F` | IoU threshold for matching (default: 0.5) |
| `--nms-thresh F` | NMS threshold (default: 0.6) |

## Architecture

```
Frozen DINOv3 Backbone
        |
   Detection Head
   ├── FPN Neck (bidirectional PAN)
   ├── Cls Tower (decoupled, cosine classifier)
   └── Reg Tower (LTRB + optional angle)
```

## Structure

```
src/
  blocks.py              # Shared building blocks (ConvGNReLU, RepDWSBlock, ECA)
  neck.py                # MultiLevelEfficientPAN + DetectHeadV3
  backbone_vitsplus.py   # DinoBackboneV3 (ViT-S/16+ wrapper)
  backbone_convnext.py   # DinoBackboneConvNeXt wrapper
  head_vitsplus_mix.py   # 3-level dual-classifier head
  head_vitsplus_mix_p2.py # 4-level P2 head + AIFI + AuxDecoder
  head_convnext.py       # ConvNeXt-specific FPN + head
  loss.py                # VFL + DFL + STAL + aux decoder loss
  decode.py              # Decode + NMS (AABB + OBB)
  dataset.py             # DatasetCOCOv3 (mosaic, copy-paste, SAHI)
  dataset_eval.py        # DatasetCOCO (simple eval loader)
  optimizer.py           # MuSGD (Muon+SGD hybrid)
config/                  # Per-pipeline hyperparameters
train/                   # Training scripts (4-stage curriculum)
eval/                    # COCO-style evaluation
inference/               # Batch video inference
```

## Configuration

All hyperparameters live in `config/config_<pipeline>.py`. Configs are loaded by file path via `importlib` (`--config config/config_vitsplus_mix_p2.py`).

## Training Outputs

Each run creates `results/<timestamp>[_<run-name>]/` with checkpoints (`best.pth`/`last.pth`) and `log.json` (per-epoch metrics).
