#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mix.py — 4-Stage Mixed COCO + Neubie Training
=====================================================

Implements the mixed-dataset strategy from mixed_dataset_training_strategy.md:

  Stage 1 (epoch   1– 50): COCO only — shared head + COCO cls warmup
  Stage 2 (epoch  51– 70): Neubie only — Neubie cls warmup (low LR on shared)
  Stage 3 (epoch  71–230): Mixed COCO + Neubie — shifting 60/40 → 40/60 → 20/80
  Stage 4 (epoch 231–250): Neubie only — final calibration

Key design points:
  - DINODetectionHeadMix: dual classifiers (COCO 80-cls + Neubie cls)
  - Loss masking: COCO batches → COCO cls loss only; Neubie batches → Neubie cls only
  - AdamW with per-component LRs (shared head vs classifiers)
  - RepeatFactorSampler for Neubie (class-balanced oversampling)
  - EMA with tau warmup (decay=0.9999)
  - Mosaic disabled for last CLOSE_MOSAIC_EPOCHS epochs
  - Neubie val set used throughout for final-class monitoring

Run
---
    nohup python -u train/train_mix.py \\
        --config config/config_vitsplus_mix_p2.py \\
        --device cuda:1 \\
        --use-amp > nohup_mix.out 2>&1 &

Resume
------
    python train/train_mix.py --config config/config_vitsplus_mix_p2.py \\
        --resume results/<run_dir>/last.pth
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── EMA (same as train_v3.py) ─────────────────────────────────────────────────

class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, tau: int = 2000):
        self.ema     = copy.deepcopy(model).eval()
        self.updates = 0
        self._decay  = decay
        self._tau    = tau
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _d(self) -> float:
        return self._decay * (1.0 - math.exp(-self.updates / self._tau))

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.updates += 1
        d = self._d()
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_((1.0 - d) * msd[k].detach())
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, sd):
        self.ema.load_state_dict(sd)

    def full_state(self) -> dict:
        """Save full EMA state including updates counter."""
        return {'ema_weights': self.ema.state_dict(), 'updates': self.updates}

    def load_full_state(self, state: dict):
        """Restore full EMA state including updates counter."""
        self.ema.load_state_dict(state['ema_weights'])
        self.updates = int(state.get('updates', 0))


# ── Stage helpers ─────────────────────────────────────────────────────────────

def get_stage(epoch: int, cfg) -> int:
    """Returns 1–4 based on 0-indexed epoch."""
    if epoch < cfg.STAGE1_END:
        return 1
    elif epoch < cfg.STAGE2_END:
        return 2
    elif epoch < cfg.STAGE3_END:
        return 3
    else:
        return 4


def get_coco_ratio(epoch: int, cfg) -> float:
    """Fraction of batches to draw from COCO in stage 3."""
    if epoch < cfg.STAGE3_SUB1_END:   # 70–109 → 60% COCO
        return 0.6
    elif epoch < cfg.STAGE3_SUB2_END:  # 110–144 → 40% COCO
        return 0.4
    else:                               # 145–179 → 20% COCO
        return 0.2


# ── RepeatFactorSampler ───────────────────────────────────────────────────────

def get_class_counts_fast(dataset):
    class_counter   = Counter()
    image_class_map = {}
    for idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns    = dataset.coco.loadAnns(ann_ids)
        cls_set = set()
        for ann in anns:
            if ann.get('iscrowd', 0):
                continue
            cat_id = ann['category_id']
            if cat_id in dataset.catid_to_label:
                cls_set.add(dataset.catid_to_label[cat_id])
        image_class_map[idx] = cls_set
        class_counter.update(cls_set)
    return dict(class_counter), image_class_map


class RepeatFactorSampler(torch.utils.data.Sampler):
    def __init__(self, image_class_map, class_counts, num_images, t=0.005, max_factor=4.0):
        self.num_images = num_images
        freq = {c: n / num_images for c, n in class_counts.items()}
        self.rep_factors = []
        for idx in range(num_images):
            cls_set = image_class_map.get(idx, set())
            if not cls_set:
                self.rep_factors.append(1.0)
            else:
                r = min(max_factor, max(max(1.0, math.sqrt(t / max(freq.get(c, 1.0), 1e-8)))
                        for c in cls_set))
                self.rep_factors.append(r)
        self._int_part  = [int(r) for r in self.rep_factors]
        self._frac_part = [r - int(r) for r in self.rep_factors]

    def __iter__(self):
        indices = []
        for idx in range(self.num_images):
            indices.extend([idx] * self._int_part[idx])
            if torch.rand(1).item() < self._frac_part[idx]:
                indices.append(idx)
        perm = torch.randperm(len(indices))
        return iter([indices[i] for i in perm.tolist()])

    def __len__(self):
        return int(sum(math.ceil(r) for r in self.rep_factors))


def make_effective_number_weights(class_counts, num_classes, beta=0.9995, max_weight=5.0):
    weights = []
    for c in range(num_classes):
        n   = class_counts.get(c, 1)
        e_n = (1.0 - beta ** n) / (1.0 - beta)
        weights.append(1.0 / max(e_n, 1e-6))
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()
    return w.clamp(min=0.25, max=max_weight)


# ── Mixed batch iterator (stage 3) ────────────────────────────────────────────

class MixedBatchIterator:
    """Interleaves COCO and Neubie batches with enforced ratio.

    Pre-computes a shuffled schedule so the realized ratio matches coco_ratio.
    The over-represented dataset is subsampled; the under-represented one is
    used in full. This guarantees the training sees the correct mix proportion.

    Yields (batch, dataset_tag) where dataset_tag is 'coco' or 'neubie'.
    """

    def __init__(self, coco_loader: DataLoader, neubie_loader: DataLoader,
                 coco_ratio: float):
        self.coco_loader   = coco_loader
        self.neubie_loader = neubie_loader
        self.coco_ratio    = coco_ratio

    def _compute_targets(self):
        n_coco   = len(self.coco_loader)
        n_neubie = len(self.neubie_loader)
        r = max(min(self.coco_ratio, 0.999), 0.001)
        # Max total steps constrained by each dataset at the target ratio
        max_by_coco   = n_coco / r
        max_by_neubie = n_neubie / (1.0 - r)
        effective_total = int(min(max_by_coco, max_by_neubie))
        target_coco   = min(int(round(r * effective_total)), n_coco)
        target_neubie = min(effective_total - target_coco, n_neubie)
        return target_coco, target_neubie

    def __iter__(self) -> Iterator[Tuple[Any, str]]:
        target_coco, target_neubie = self._compute_targets()

        # Build a shuffled schedule
        schedule: List[str] = ['coco'] * target_coco + ['neubie'] * target_neubie
        random.shuffle(schedule)

        coco_iter   = iter(self.coco_loader)
        neubie_iter = iter(self.neubie_loader)

        for tag in schedule:
            try:
                if tag == 'coco':
                    yield next(coco_iter), 'coco'
                else:
                    yield next(neubie_iter), 'neubie'
            except StopIteration:
                break

    def __len__(self):
        target_coco, target_neubie = self._compute_targets()
        return target_coco + target_neubie


# ── Utility helpers ───────────────────────────────────────────────────────────

def load_cfg_from_py(py_path: str):
    py_path = str(Path(py_path).resolve())
    spec = importlib.util.spec_from_file_location('user_cfg', py_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_run_dir(results_path, dino_model_name, run_name=None):
    ts   = _dt.datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_{dino_model_name}_mix')
    name = run_name if run_name else ts
    out  = os.path.join(results_path, name)
    os.makedirs(out, exist_ok=True)
    return out


def collect_cfg_dict(args, cfg_mod):
    base: Dict[str, Any] = {}
    for k in dir(cfg_mod):
        if k.isupper():
            v = getattr(cfg_mod, k)
            if isinstance(v, (int, float, str, bool, list, dict, tuple)) or v is None:
                base[k] = v
    base.update({f'CLI_{k.upper()}': getattr(args, k) for k in vars(args)})
    return base


def compute_strides(images, outputs):
    img_w  = images.shape[-1]
    feat_w = outputs['cls'][0].shape[3]
    s0     = float(img_w / feat_w)
    return [s0 * (2 ** l) for l in range(len(outputs['cls']))]


def save_checkpoint(path, epoch, model_head, dino_backbone, optimizer, cfg_dict,
                    best_val_loss=None, save_backbone=False, scaler_state=None,
                    ema_state=None, ema_full_state=None):
    ckpt: Dict[str, Any] = {
        'epoch':         epoch,
        'model_head':    model_head.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'cfg':           cfg_dict,
        'best_val_loss': best_val_loss,
        'scaler':        scaler_state,
    }
    if ema_full_state is not None:
        ckpt['ema_full'] = ema_full_state
    if ema_state is not None:
        ckpt['ema'] = ema_state
    if save_backbone:
        ckpt['dino_backbone'] = dino_backbone.state_dict()
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model_head, dino_backbone, optimizer=None, ema=None,
                    map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
    # Try strict load first; fall back to shape-filtered partial load
    # (needed when resuming from a pre-Phase-C checkpoint into a Phase-C model)
    try:
        model_head.load_state_dict(sd)
    except RuntimeError:
        model_sd = model_head.state_dict()
        loaded = {k: v for k, v in sd.items()
                  if k in model_sd and model_sd[k].shape == v.shape}
        fresh = [k for k in model_sd if k not in loaded]
        skipped = [k for k in sd if k not in model_sd or
                   (k in model_sd and model_sd[k].shape != sd[k].shape)]
        model_sd.update(loaded)
        model_head.load_state_dict(model_sd, strict=True)
        print(f'  [resume partial-load] loaded={len(loaded)}/{len(sd)} '
              f'fresh={len(fresh)} skipped={len(skipped)}')
        if fresh:
            print(f'    fresh (new modules): {fresh[:8]}{"..." if len(fresh)>8 else ""}')
    if isinstance(ckpt, dict) and 'dino_backbone' in ckpt:
        dino_backbone.load_state_dict(ckpt['dino_backbone'])
    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError):
            print('  [resume] optimizer state incompatible (architecture changed) — '
                  'starting optimizer fresh')
    # Restore EMA: prefer full state (with updates counter), fall back to weights-only
    ema_restored = False
    if ema is not None and isinstance(ckpt, dict):
        if 'ema_full' in ckpt:
            ema.load_full_state(ckpt['ema_full'])
            ema_restored = True
        elif 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
            # Legacy checkpoint without updates counter — estimate from epoch
            epoch_val = int(ckpt.get('epoch', 0))
            ema.updates = max(ema.updates, epoch_val * 500)  # conservative estimate
            ema_restored = True
    start_epoch  = int(ckpt.get('epoch', -1)) + 1 if isinstance(ckpt, dict) else 0
    best_val     = ckpt.get('best_val_loss', None) if isinstance(ckpt, dict) else None
    scaler_state = ckpt.get('scaler', None)         if isinstance(ckpt, dict) else None
    return start_epoch, best_val, scaler_state, ema_restored


# ── Validation helpers ────────────────────────────────────────────────────────

def box_iou_matrix(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0], device=boxes1.device)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(0))
    lt    = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb    = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh    = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    return inter / union


def update_det_metrics(class_stats, pred_boxes, pred_scores, pred_labels,
                       gt_boxes, gt_labels, iou_thresh=0.5):
    all_cls = (torch.unique(torch.cat([pred_labels, gt_labels]))
               if (pred_labels.numel() or gt_labels.numel())
               else torch.empty(0, dtype=torch.long))
    for cls in all_cls.tolist():
        pm = pred_labels == cls;  gm = gt_labels == cls
        p_b = pred_boxes[pm];  p_s = pred_scores[pm];  g_b = gt_boxes[gm]
        class_stats[cls]['gt'] += int(g_b.shape[0])
        if p_s.numel() > 0:
            class_stats[cls]['score_sum']   += float(p_s.sum())
            class_stats[cls]['score_count'] += int(p_s.numel())
        if p_b.numel() == 0:
            continue
        if g_b.numel() == 0:
            class_stats[cls]['fp'] += int(p_b.shape[0])
            continue
        order   = torch.argsort(p_s, descending=True)
        p_b     = p_b[order];  p_s = p_s[order]
        ious    = box_iou_matrix(p_b, g_b)
        matched = torch.zeros(g_b.shape[0], dtype=torch.bool, device=g_b.device)
        for i in range(p_b.shape[0]):
            bv, bj = ious[i].max(0)
            if bv >= iou_thresh and not matched[bj]:
                matched[bj] = True
                class_stats[cls]['tp'] += 1
            else:
                class_stats[cls]['fp'] += 1
        class_stats[cls]['fn'] += int((~matched).sum())


def abs_gt_boxes(boxes_rel, img_h, img_w):
    boxes_rel = boxes_rel.reshape(-1, 4)
    if boxes_rel.numel() == 0:
        return boxes_rel.new_zeros((0, 4))
    x1 = boxes_rel[:, 0] * img_w;  y1 = boxes_rel[:, 1] * img_h
    x2 = x1 + boxes_rel[:, 2] * img_w
    y2 = y1 + boxes_rel[:, 3] * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywhr2xyxy_aabb(boxes_xywhr):
    if boxes_xywhr.shape[0] == 0:
        return boxes_xywhr.new_zeros((0, 4))
    from src.decode import xywhr2xyxyxyxy
    corners = xywhr2xyxyxyxy(boxes_xywhr)
    x1, _ = corners[:, :, 0].min(1);  y1, _ = corners[:, :, 1].min(1)
    x2, _ = corners[:, :, 0].max(1);  y2, _ = corners[:, :, 1].max(1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def summarize_metrics(class_stats, class_names, focus_names):
    lines = []
    for cid, name in enumerate(class_names):
        if name not in focus_names:
            continue
        st = class_stats.get(cid, {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0,
                                    'score_sum': 0.0, 'score_count': 0})
        p  = st['tp'] / max(1, st['tp'] + st['fp'])
        r  = st['tp'] / max(1, st['tp'] + st['fn'])
        av = st['score_sum'] / max(1, st['score_count'])
        lines.append(f'{name}: P={p:.3f} R={r:.3f} GT={st["gt"]} avgScore={av:.3f}')
    return ' | '.join(lines) if lines else 'No focus-class stats.'


# ── LR helpers ────────────────────────────────────────────────────────────────

def set_lr(optimizer, group_name: str, new_lr: float):
    """Update learning rate for a named param group."""
    for pg in optimizer.param_groups:
        if pg.get('name') == group_name:
            pg['lr'] = new_lr


def adjust_lr_for_stage(optimizer, stage: int, cfg):
    """Apply stage-specific LR rules."""
    if stage == 2:
        # Shared head gets low LR so it doesn't overwrite COCO-learned features
        set_lr(optimizer, 'shared',     cfg.LR_SHARED_STAGE2)
        set_lr(optimizer, 'cls_coco',   0.0)           # COCO cls not used in stage 2
        set_lr(optimizer, 'cls_neubie', cfg.LR_CLS_NEUBIE)
    elif stage == 1:
        set_lr(optimizer, 'shared',     cfg.LR_SHARED)
        set_lr(optimizer, 'cls_coco',   cfg.LR_CLS_COCO)
        set_lr(optimizer, 'cls_neubie', 0.0)           # Neubie cls not used in stage 1
    else:
        # Stages 3 and 4: all components active
        set_lr(optimizer, 'shared',     cfg.LR_SHARED)
        set_lr(optimizer, 'cls_coco',   cfg.LR_CLS_COCO)
        set_lr(optimizer, 'cls_neubie', cfg.LR_CLS_NEUBIE)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    p = argparse.ArgumentParser('DINOv3 mixed COCO+Neubie training')
    p.add_argument('--config',             type=str,  default='')
    p.add_argument('--run-name',           type=str,  default='')
    p.add_argument('--device',             type=str,  default='')
    p.add_argument('--num-workers',        type=int,  default=8)
    p.add_argument('--pin-memory',         action='store_true')
    p.add_argument('--persistent-workers', action='store_true')
    p.add_argument('--prefetch-factor',    type=int,  default=4)
    p.add_argument('--batch-size',         type=int,  default=-1)
    p.add_argument('--epochs',             type=int,  default=-1)
    p.add_argument('--use-amp',            action='store_true')
    p.add_argument('--save-every-steps',   type=int,  default=0)
    p.add_argument('--resume',             type=str,  default='')
    p.add_argument('--patience',           type=int,  default=30)
    p.add_argument('--save-every-epochs',  type=int,  default=10,
                   help='Save model_<epoch>.pth every N epochs (0 = disable)')
    p.add_argument('--warm-start',         type=str,  default='',
                   help='3-level best_e* checkpoint to partial-load into the 4-level '
                        'P2 model (P3→P2 clone). Ignored if --resume is given.')
    p.add_argument('--unfreeze-last-n',    type=int,  default=-1,
                   help='Unfreeze the last N DINOv3 blocks during Stage 3 '
                        '(0 = frozen/Phase 2, 2 = Phase 3). -1 = use config UNFREEZE_LAST_N.')
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    if not args.config:
        raise ValueError('--config is required')
    cfg = load_cfg_from_py(args.config)

    from src.dataset   import DatasetCOCOv3
    from src.backbone_vitsplus import DinoBackboneV3
    from src.head_vitsplus_mix_p2 import DINODetectionHeadMixP2, warm_start_from_3level
    from src.loss           import compute_loss
    from src.decode         import decode_outputs_OBB
    from collate_fn            import collate_fn

    device = (torch.device(args.device) if args.device
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Using device:', device)

    cfg_use_amp = bool(getattr(cfg, 'USE_AMP', False))
    use_amp     = (args.use_amp or cfg_use_amp) and device.type == 'cuda'
    scaler      = (GradScaler('cuda', enabled=use_amp) if device.type == 'cuda'
                   else GradScaler('cpu', enabled=False))
    print('AMP enabled:', use_amp)

    # ── Config ────────────────────────────────────────────────────────────────
    NEUBIE_ROOT       = cfg.NEUBIE_ROOT
    COCO_ROOT_MIX     = cfg.COCO_ROOT_MIX
    IMG_SIZE          = cfg.IMG_SIZE
    PATCH_SIZE        = cfg.PATCH_SIZE
    IMG_MEAN          = cfg.IMG_MEAN
    IMG_STD           = cfg.IMG_STD
    PROB_AUG_TRAIN    = cfg.PROB_AUGMENT_TRAINING
    PROB_AUG_VAL      = cfg.PROB_AUGMENT_VALID
    DINOV3_DIR        = cfg.DINOV3_DIR
    DINO_MODEL        = cfg.DINO_MODEL
    DINO_WEIGHTS      = cfg.DINO_WEIGHTS
    FPN_CH            = cfg.FPN_CH
    N_CONVS           = cfg.N_CONVS
    NUM_COCO_CLASSES  = cfg.NUM_COCO_CLASSES
    NUM_NEUBIE_CLASSES = cfg.NUM_NEUBIE_CLASSES
    BATCH_SIZE        = cfg.BATCH_SIZE if args.batch_size < 0 else args.batch_size
    NUM_EPOCHS        = cfg.NUM_EPOCHS if args.epochs < 0 else args.epochs
    WEIGHT_REG        = cfg.WEIGHT_REG
    WEIGHT_CTR        = cfg.WEIGHT_CTR
    WEIGHT_ANGLE      = float(getattr(cfg, 'ANGLE_WEIGHT', 0.2))
    WEIGHT_DECAY      = cfg.WEIGHT_DECAY
    LR_WARMUP_EPOCHS  = int(getattr(cfg, 'LR_WARMUP_EPOCHS', 5))
    LR_MIN            = float(getattr(cfg, 'LR_MIN', 1e-6))
    FOCAL_ALPHA       = cfg.FOCAL_ALPHA
    FOCAL_GAMMA       = cfg.FOCAL_GAMMA
    VFL_ALPHA         = float(getattr(cfg, 'VFL_ALPHA', 0.75))
    VFL_CW_NEG        = bool(getattr(cfg, 'VFL_CW_NEG', False))
    VFL_Q_FLOOR       = float(getattr(cfg, 'VFL_Q_FLOOR', 0.0))
    USE_OBB           = bool(getattr(cfg, 'USE_OBB', True))
    BEST_METRIC       = str(getattr(cfg, 'BEST_METRIC', 'f1')).lower()
    TAL_TOPK          = int(getattr(cfg, 'TAL_TOPK', 12))
    TAL_ALPHA         = float(getattr(cfg, 'TAL_ALPHA', 0.5))
    TAL_BETA          = float(getattr(cfg, 'TAL_BETA', 4.0))
    PROG_LOSS_EPOCHS  = int(getattr(cfg, 'PROG_LOSS_EPOCHS', 10))
    USE_EMA           = bool(getattr(cfg, 'USE_EMA', True))
    EMA_DECAY         = float(getattr(cfg, 'EMA_DECAY', 0.9999))
    EMA_TAU           = int(getattr(cfg, 'EMA_TAU', 2000))
    USE_MOSAIC        = bool(getattr(cfg, 'USE_MOSAIC', True))
    MOSAIC_PROB       = float(getattr(cfg, 'MOSAIC_PROB', 0.5))
    CLOSE_MOSAIC      = int(getattr(cfg, 'CLOSE_MOSAIC_EPOCHS', 10))
    HFLIP_PROB        = float(getattr(cfg, 'HFLIP_PROB', 0.5))
    TRANSLATE_PROB    = float(getattr(cfg, 'TRANSLATE_PROB', 0.5))
    TRANSLATE_MAX     = float(getattr(cfg, 'TRANSLATE_MAX', 0.1))
    SAVE_MODEL        = bool(getattr(cfg, 'SAVE_MODEL', True))
    RESULTS_PATH      = cfg.RESULTS_PATH
    VAL_IOU_THRESH    = float(getattr(cfg, 'VAL_IOU_THRESH', 0.5))
    VAL_SCORE_THRESH  = float(getattr(cfg, 'VAL_SCORE_THRESH', 0.2))
    VAL_NMS_THRESH    = float(getattr(cfg, 'VAL_NMS_THRESH', 0.6))
    VAL_RARE_THRESH   = float(getattr(cfg, 'VAL_RARE_THRESH', 0.08))
    VAL_METRIC_EVERY  = int(getattr(cfg, 'VAL_METRIC_EVERY', 1))
    # P2 / unfreeze knobs
    UNFREEZE_LAST_N   = (args.unfreeze_last_n if args.unfreeze_last_n >= 0
                         else int(getattr(cfg, 'UNFREEZE_LAST_N', 0)))
    LR_BACKBONE       = float(getattr(cfg, 'LR_BACKBONE', 1e-5))
    BACKBONE_WD       = float(getattr(cfg, 'BACKBONE_WD', 1e-4))
    WARM_START_CKPT   = args.warm_start or str(getattr(cfg, 'WARM_START_CKPT', ''))

    # Phase C architectural knobs
    USE_AIFI          = bool(getattr(cfg, 'USE_AIFI', False))
    AIFI_LAYERS       = int(getattr(cfg, 'AIFI_LAYERS', 2))
    AIFI_HEADS        = int(getattr(cfg, 'AIFI_HEADS', 8))
    USE_DFL           = bool(getattr(cfg, 'USE_DFL', False))
    DFL_REG_MAX       = int(getattr(cfg, 'DFL_REG_MAX', 16)) if USE_DFL else 0
    USE_CENTERNESS    = bool(getattr(cfg, 'USE_CENTERNESS', True))
    USE_DCN           = bool(getattr(cfg, 'USE_DCN', False))
    USE_AUX_DECODER   = bool(getattr(cfg, 'USE_AUX_DECODER', False))
    AUX_NUM_QUERIES   = int(getattr(cfg, 'AUX_NUM_QUERIES', 100))
    AUX_DECODER_LAYERS = int(getattr(cfg, 'AUX_DECODER_LAYERS', 2))
    AUX_LOSS_WEIGHT   = float(getattr(cfg, 'AUX_LOSS_WEIGHT', 0.5))

    n_workers_kw = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Small-object augmentation knobs (default OFF if absent → backward compatible)
    HUE_DELTA             = float(getattr(cfg, 'HUE_DELTA', 10))
    SATURATION_RANGE      = tuple(getattr(cfg, 'SATURATION_RANGE', (0.8, 1.2)))
    ZOOMOUT_PROB          = float(getattr(cfg, 'ZOOMOUT_PROB', 0.0))
    ZOOMOUT_MAX_SCALE     = float(getattr(cfg, 'ZOOMOUT_MAX_SCALE', 2.0))
    COPYPASTE_PROB        = float(getattr(cfg, 'COPYPASTE_PROB', 0.0))
    COPYPASTE_MAX_PER_IMG = int(getattr(cfg, 'COPYPASTE_MAX_PER_IMG', 3))
    COPYPASTE_CLASSES     = tuple(getattr(cfg, 'COPYPASTE_CLASSES', ()))
    COPYPASTE_SCALE_RANGE = tuple(getattr(cfg, 'COPYPASTE_SCALE_RANGE', (0.6, 1.4)))
    SAHICROP_PROB         = float(getattr(cfg, 'SAHICROP_PROB', 0.0))
    SAHICROP_UPPER_FRAC   = float(getattr(cfg, 'SAHICROP_UPPER_FRAC', 0.55))
    SAHICROP_COLS         = int(getattr(cfg, 'SAHICROP_COLS', 2))
    SAHICROP_OVERLAP      = float(getattr(cfg, 'SAHICROP_OVERLAP', 0.2))
    SAHICROP_MIN_VIS      = float(getattr(cfg, 'SAHICROP_MIN_VIS', 0.3))

    # ── Neubie dataset ────────────────────────────────────────────────────────
    # Geometric + scale + SAHI-crop knobs are shared by both datasets (copy-paste
    # is a no-op on COCO since it has none of the target Neubie class names).
    aug_kw = dict(
        mosaic_prob=MOSAIC_PROB, hflip_prob=HFLIP_PROB,
        translate_prob=TRANSLATE_PROB, translate_max=TRANSLATE_MAX,
        hue_delta=HUE_DELTA, saturation_range=SATURATION_RANGE,
        zoomout_prob=ZOOMOUT_PROB, zoomout_max_scale=ZOOMOUT_MAX_SCALE,
        copypaste_prob=COPYPASTE_PROB, copypaste_max_per_img=COPYPASTE_MAX_PER_IMG,
        copypaste_classes=COPYPASTE_CLASSES, copypaste_scale_range=COPYPASTE_SCALE_RANGE,
        sahicrop_prob=SAHICROP_PROB, sahicrop_upper_frac=SAHICROP_UPPER_FRAC,
        sahicrop_cols=SAHICROP_COLS, sahicrop_overlap=SAHICROP_OVERLAP,
        sahicrop_min_vis=SAHICROP_MIN_VIS,
    )
    neubie_train = DatasetCOCOv3(
        NEUBIE_ROOT, 'train', IMG_SIZE, PATCH_SIZE,
        augment_prob=PROB_AUG_TRAIN, mean=IMG_MEAN, std=IMG_STD,
        layout='custom', use_mosaic=USE_MOSAIC, **aug_kw,
    )
    neubie_val = DatasetCOCOv3(
        NEUBIE_ROOT, 'val', IMG_SIZE, PATCH_SIZE,
        augment_prob=PROB_AUG_VAL, mean=IMG_MEAN, std=IMG_STD,
        layout='custom', use_mosaic=False,
    )

    print('Building Neubie class counts …')
    neubie_counts, neubie_img_cls_map = get_class_counts_fast(neubie_train)
    neubie_sampler = RepeatFactorSampler(
        neubie_img_cls_map, neubie_counts, len(neubie_train))
    print(f'  Neubie RepeatFactorSampler: {len(neubie_sampler)} samples/epoch '
          f'(dataset: {len(neubie_train)})')

    neubie_class_weights = make_effective_number_weights(
        neubie_counts, len(neubie_train.class_names)).to(device)

    neubie_train_loader = DataLoader(
        neubie_train, batch_size=BATCH_SIZE, sampler=neubie_sampler,
        shuffle=False, collate_fn=collate_fn, **n_workers_kw)
    neubie_val_loader = DataLoader(
        neubie_val, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, **n_workers_kw)

    # ── COCO dataset ──────────────────────────────────────────────────────────
    coco_train = DatasetCOCOv3(
        COCO_ROOT_MIX, 'train', IMG_SIZE, PATCH_SIZE,
        augment_prob=PROB_AUG_TRAIN, mean=IMG_MEAN, std=IMG_STD,
        layout='coco', use_mosaic=USE_MOSAIC, **aug_kw,
    )

    # Sanity check COCO class count
    assert len(coco_train.class_names) == NUM_COCO_CLASSES, (
        f'Expected {NUM_COCO_CLASSES} COCO classes, got {len(coco_train.class_names)}')

    print('Building COCO class counts …')
    coco_counts, coco_img_cls_map = get_class_counts_fast(coco_train)
    coco_sampler = RepeatFactorSampler(
        coco_img_cls_map, coco_counts, len(coco_train))
    print(f'  COCO RepeatFactorSampler: {len(coco_sampler)} samples/epoch '
          f'(dataset: {len(coco_train)})')

    coco_class_weights = make_effective_number_weights(
        coco_counts, len(coco_train.class_names)).to(device)

    coco_train_loader = DataLoader(
        coco_train, batch_size=BATCH_SIZE, sampler=coco_sampler,
        shuffle=False, collate_fn=collate_fn, **n_workers_kw)

    print(f'Neubie classes ({NUM_NEUBIE_CLASSES}): '
          f'{neubie_train.class_names}')
    print(f'COCO classes  ({NUM_COCO_CLASSES}): '
          f'{coco_train.class_names[:10]} ... (first 10)')

    # ── Model ─────────────────────────────────────────────────────────────────
    n_layers  = cfg.MODEL_TO_NUM_LAYERS[DINO_MODEL]
    embed_dim = cfg.MODEL_TO_EMBED_DIM[DINO_MODEL]

    dino_model    = torch.hub.load(repo_or_dir=DINOV3_DIR, model=DINO_MODEL,
                                   source='local', weights=DINO_WEIGHTS)
    dino_backbone = DinoBackboneV3(dino_model, n_layers).to(device)
    # Derive actual class counts from loaded datasets (avoids config drift)
    actual_neubie_classes = len(neubie_train.class_names)
    actual_coco_classes   = len(coco_train.class_names)
    assert actual_neubie_classes == NUM_NEUBIE_CLASSES, (
        f'Config NUM_NEUBIE_CLASSES={NUM_NEUBIE_CLASSES} but dataset has '
        f'{actual_neubie_classes} classes. Fix config_mix.py.')
    assert actual_coco_classes == NUM_COCO_CLASSES, (
        f'Config NUM_COCO_CLASSES={NUM_COCO_CLASSES} but dataset has '
        f'{actual_coco_classes} classes. Fix config_mix.py.')

    model_head    = DINODetectionHeadMixP2(
        backbone_out_channels=embed_dim,
        fpn_channels=FPN_CH,
        num_coco_classes=actual_coco_classes,
        num_neubie_classes=actual_neubie_classes,
        num_convs=N_CONVS,
        obb=USE_OBB,
        # Phase C
        use_aifi=USE_AIFI,
        aifi_layers=AIFI_LAYERS,
        aifi_heads=AIFI_HEADS,
        reg_max=DFL_REG_MAX,
        use_centerness=USE_CENTERNESS,
        use_dcn=USE_DCN,
        use_aux_decoder=USE_AUX_DECODER,
        aux_num_queries=AUX_NUM_QUERIES,
        aux_decoder_layers=AUX_DECODER_LAYERS,
    ).to(device)
    print(f'P2 (stride-8) head: 4-level pyramid [P2,P3,P4,P5]  '
          f'OBB={"ENABLED" if USE_OBB else "DISABLED"}')
    print(f'  AIFI={USE_AIFI} DFL={USE_DFL}(reg_max={DFL_REG_MAX}) '
          f'CTR={USE_CENTERNESS} DCN={USE_DCN} AUX={USE_AUX_DECODER}')

    # ── Warm-start (partial-load from a checkpoint into this model) ─────────────
    # Handles both:
    #   1) 3-level → 4-level P2 remap (warm_start_from_3level)
    #   2) Same-architecture but Phase C shape mismatches (shape-filtered partial load)
    # Only when not resuming a P2 run of our own.
    if WARM_START_CKPT and not args.resume:
        rep = warm_start_from_3level(model_head, WARM_START_CKPT, map_location='cpu')
        print(f'  [warm-start] from {WARM_START_CKPT}')
        print(f'    loaded={rep["loaded"]} fresh_init={rep["fresh_init"]} '
              f'shape_mismatch={rep["shape_mismatch"]} '
              f'unexpected={rep["unexpected"]}')
        if rep['fresh_keys']:
            print(f'    fresh keys (expected): {rep["fresh_keys"][:10]}...'
                  if len(rep['fresh_keys']) > 10 else
                  f'    fresh keys: {rep["fresh_keys"]}')
        # Shape mismatches are expected with Phase C (DFL bbox_reg, DCN cls_tower, etc.)
        if rep['unexpected']:
            print(f'    [WARN] unexpected keys: {rep["unexpected_keys"][:5]}')

    # ── Backbone freeze / partial-unfreeze ──────────────────────────────────────
    # Default: fully frozen (Phase 2). With UNFREEZE_LAST_N>0 (Phase 3) the last N
    # ViT blocks become trainable, but ONLY during Stage 3 (toggled in the loop).
    # eval() is kept always (disables DINOv3 stochastic depth — deterministic).
    def set_backbone_trainable(n_last: int):
        for p_ in dino_backbone.parameters():
            p_.requires_grad = False
        if n_last > 0:
            blocks = dino_backbone.dino.blocks
            for blk in list(blocks)[-n_last:]:
                for p_ in blk.parameters():
                    p_.requires_grad = True

    def backbone_trainable_param_groups():
        """Split currently-trainable backbone params into (weights, norm/bias)."""
        weights, nwd = [], []
        for name, p_ in dino_backbone.named_parameters():
            if not p_.requires_grad:
                continue
            (nwd if (p_.ndim == 1 or name.endswith('.bias')) else weights).append(p_)
        return weights, nwd

    set_backbone_trainable(0)          # start frozen; Stage 3 will unfreeze if enabled
    dino_backbone.eval()
    if UNFREEZE_LAST_N > 0:
        print(f'  [unfreeze] Phase 3: last {UNFREEZE_LAST_N} ViT blocks trainable in '
              f'Stage 3 only (LR_BACKBONE={LR_BACKBONE}, WD weights={BACKBONE_WD}/norm-bias=0)')
    else:
        print('  [unfreeze] Phase 2: backbone fully frozen')

    total_params = sum(p_.numel() for p_ in model_head.parameters())
    shared_count = sum(p_.numel() for p_ in model_head.shared_params())
    print(f'Head params: {total_params/1e6:.2f} M  '
          f'(shared={shared_count/1e6:.2f} M, '
          f'cls_coco={sum(p.numel() for p in model_head.coco_cls_params())/1e6:.2f} M, '
          f'cls_neubie={sum(p.numel() for p in model_head.neubie_cls_params())/1e6:.2f} M)')

    # ── EMA ───────────────────────────────────────────────────────────────────
    ema: Optional[ModelEMA] = None
    ema_backbone: Optional[ModelEMA] = None
    if USE_EMA:
        ema = ModelEMA(model_head, decay=EMA_DECAY, tau=EMA_TAU)
        print(f'EMA enabled (head): decay={EMA_DECAY}, tau={EMA_TAU}')
        if UNFREEZE_LAST_N > 0:
            # Phase 3: the backbone trains, so validation must use the EMA backbone
            # together with the EMA head (else val mixes EMA-head + live-backbone).
            ema_backbone = ModelEMA(dino_backbone, decay=EMA_DECAY, tau=EMA_TAU)
            print('EMA enabled (backbone) — Phase 3')

    # ── AdamW with per-component LR ───────────────────────────────────────────
    # Separate param groups allow per-stage LR adjustments without rebuilding optimizer
    CLS_WEIGHT_DECAY = float(getattr(cfg, 'CLS_WEIGHT_DECAY', 1e-5))
    GRAD_CLIP_NORM   = float(getattr(cfg, 'GRAD_CLIP_NORM', 5.0))
    if UNFREEZE_LAST_N > 0:
        GRAD_CLIP_NORM = min(GRAD_CLIP_NORM, 0.1)   # tighter clip for backbone grads
        print(f'  [unfreeze] grad-clip tightened to {GRAD_CLIP_NORM}')

    opt_groups = [
        {'params': model_head.shared_params(),     'lr': cfg.LR_SHARED,
         'name': 'shared',     'weight_decay': WEIGHT_DECAY},
        {'params': model_head.coco_cls_params(),   'lr': cfg.LR_CLS_COCO,
         'name': 'cls_coco',   'weight_decay': CLS_WEIGHT_DECAY},
        {'params': model_head.neubie_cls_params(), 'lr': cfg.LR_CLS_NEUBIE,
         'name': 'cls_neubie', 'weight_decay': CLS_WEIGHT_DECAY},
    ]
    if UNFREEZE_LAST_N > 0:
        # Make blocks trainable just to populate the param groups; the loop gates
        # the actual LR (0 outside Stage 3) and re-freezes requires_grad per stage.
        set_backbone_trainable(UNFREEZE_LAST_N)
        bb_w, bb_nwd = backbone_trainable_param_groups()
        set_backbone_trainable(0)   # back to frozen until Stage 3
        opt_groups += [
            {'params': bb_w,   'lr': 0.0, 'name': 'backbone',     'weight_decay': BACKBONE_WD},
            {'params': bb_nwd, 'lr': 0.0, 'name': 'backbone_nwd', 'weight_decay': 0.0},
        ]
    optimizer = optim.AdamW(opt_groups)
    print(f'AdamW: LR_shared={cfg.LR_SHARED}, LR_cls={cfg.LR_CLS_NEUBIE}, '
          f'CLS_WD={CLS_WEIGHT_DECAY}'
          + (f', LR_backbone={LR_BACKBONE}' if UNFREEZE_LAST_N > 0 else ''))

    # Stage peak LRs — used to compute warmup and cosine decay per stage.
    # Backbone groups are nonzero ONLY in Stage 3 (adapt with COCO present);
    # 0 elsewhere so Stage 4 calibration runs on a frozen backbone.
    STAGE_PEAK_LR = {
        'shared':     {1: cfg.LR_SHARED, 2: cfg.LR_SHARED_STAGE2,
                       3: cfg.LR_SHARED, 4: cfg.LR_SHARED},
        'cls_coco':   {1: cfg.LR_CLS_COCO, 2: 0.0,
                       3: cfg.LR_CLS_COCO, 4: cfg.LR_CLS_COCO},
        'cls_neubie': {1: 0.0, 2: cfg.LR_CLS_NEUBIE,
                       3: cfg.LR_CLS_NEUBIE, 4: cfg.LR_CLS_NEUBIE},
        'backbone':     {1: 0.0, 2: 0.0, 3: LR_BACKBONE, 4: 0.0},
        'backbone_nwd': {1: 0.0, 2: 0.0, 3: LR_BACKBONE, 4: 0.0},
    }
    # Stage boundaries (0-indexed epoch start of each stage)
    STAGE_START = {1: 0, 2: cfg.STAGE1_END, 3: cfg.STAGE2_END, 4: cfg.STAGE3_END}
    STAGE_LEN   = {1: cfg.STAGE1_END,
                   2: cfg.STAGE2_END   - cfg.STAGE1_END,
                   3: cfg.STAGE3_END   - cfg.STAGE2_END,
                   4: NUM_EPOCHS       - cfg.STAGE3_END}

    def compute_lr(group_name: str, stage: int, epoch_in_stage: int) -> float:
        """Warmup + cosine decay within a stage. Returns the LR for this epoch."""
        peak = STAGE_PEAK_LR[group_name][stage]
        if peak == 0.0:
            return 0.0
        stage_len = STAGE_LEN[stage]
        # Guard: a stage must not spend all (or most) of its epochs in warmup, or
        # it never reaches peak LR (the old 2-epoch stage 4 bug). Cap warmup so at
        # least one cosine-decay epoch remains, and never exceed half the stage.
        warmup = min(LR_WARMUP_EPOCHS, max(1, (stage_len + 1) // 2))
        # Linear warmup for first `warmup` epochs of the stage
        if epoch_in_stage < warmup:
            return LR_MIN + (peak - LR_MIN) * (epoch_in_stage + 1) / warmup
        # Cosine decay for the rest
        progress = (epoch_in_stage - warmup) / max(1, stage_len - warmup)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN + (peak - LR_MIN) * cosine

    # ── Run dir ───────────────────────────────────────────────────────────────
    run_dir  = build_run_dir(RESULTS_PATH, DINO_MODEL, args.run_name or None) if SAVE_MODEL else ''
    cfg_dict = collect_cfg_dict(args, cfg)
    if SAVE_MODEL:
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)

    start_epoch:  int   = 0
    best_val_loss: Optional[float] = None
    if args.resume:
        start_epoch, best_val_loss, scaler_state, ema_restored = load_checkpoint(
            args.resume, model_head=model_head, dino_backbone=dino_backbone,
            optimizer=optimizer, ema=ema, map_location='cpu',
        )
        if use_amp and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as e:
                print(f'[WARN] scaler state load failed: {e}')
        if ema is not None:
            if ema_restored:
                print(f'  EMA restored: updates={ema.updates}, decay={ema._d():.6f}')
            else:
                print(f'  [WARN] EMA state not found in checkpoint — using fresh EMA')
        print(f'Resumed from {args.resume} @ epoch={start_epoch}')

    global_step      = 0
    log_path         = os.path.join(run_dir, 'train_log.jsonl') if SAVE_MODEL else ''
    patience_counter = 0
    # Crucial small classes we optimize/measure explicitly (the P2 + aug levers
    # target exactly these): bollard, scooter, warning_light, traffic_light_*.
    crucial_small    = {'bollard', 'scooter', 'warning_light',
                        'traffic_light_red', 'traffic_light_green', 'traffic_light_other'}
    focus_names      = {n for n in neubie_train.class_names
                        if 'traffic_light' in n
                        or n in {'warning_light', 'neubie', 'ev_open',
                                 'bollard', 'scooter'}}
    rare_classes     = {'traffic_light_red', 'traffic_light_green',
                        'traffic_light_other', 'warning_light', 'neubie', 'ev_open',
                        'bollard', 'scooter'}
    prev_stage       = -1
    # Pre-compute trainable params once — all model_head params require grad
    trainable_params = list(model_head.parameters())

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        stage = get_stage(epoch, cfg)

        # Stage transition: print banner and reset patience
        if stage != prev_stage:
            stage_desc = {
                1: 'COCO warmup (obj + box + COCO cls)',
                2: 'Neubie cls warmup (Neubie only, low shared LR)',
                3: f'Mixed COCO+Neubie (ratio={get_coco_ratio(epoch, cfg):.0%} COCO)',
                4: 'Neubie-only calibration',
            }[stage]
            print(f'\n{"═"*68}', flush=True)
            print(f'  STAGE {stage}: epoch {epoch+1}/{NUM_EPOCHS} — {stage_desc}',
                  flush=True)
            print(f'{"═"*68}\n', flush=True)
            if prev_stage >= 0:
                patience_counter = 0
                print(f'  [Patience reset at stage {prev_stage}→{stage} transition]',
                      flush=True)
            prev_stage = stage

            # Phase 3: unfreeze last-N ViT blocks during Stage 3 only; re-freeze
            # for Stage 4 calibration (locks adapted features, avoids small-set overfit).
            if UNFREEZE_LAST_N > 0:
                set_backbone_trainable(UNFREEZE_LAST_N if stage == 3 else 0)
                dino_backbone.eval()   # always eval (no stochastic depth)
                print(f'  [backbone] {"UNFROZEN last "+str(UNFREEZE_LAST_N)+" blocks" if stage==3 else "FROZEN"}',
                      flush=True)
            # Recompute the grad-clip list (changes when backbone (un)freezes)
            trainable_params = ([p for p in model_head.parameters() if p.requires_grad] +
                                [p for p in dino_backbone.parameters() if p.requires_grad])

        # ── Per-epoch LR: warmup + cosine within stage ────────────────────────
        epoch_in_stage = epoch - STAGE_START[stage]
        current_lrs = {}
        for pg in optimizer.param_groups:
            name = pg['name']
            new_lr = compute_lr(name, stage, epoch_in_stage)
            pg['lr'] = new_lr
            current_lrs[name] = new_lr
        print(f'  LR: shared={current_lrs["shared"]:.2e}  '
              f'cls_coco={current_lrs["cls_coco"]:.2e}  '
              f'cls_neubie={current_lrs["cls_neubie"]:.2e}', flush=True)

        # ── Mosaic toggle ─────────────────────────────────────────────────────
        use_mosaic_this_epoch = USE_MOSAIC and (epoch < NUM_EPOCHS - CLOSE_MOSAIC)
        for ds in (neubie_train, coco_train):
            if ds.use_mosaic != use_mosaic_this_epoch:
                ds.use_mosaic = use_mosaic_this_epoch
        if epoch == NUM_EPOCHS - CLOSE_MOSAIC:
            print(f'  [Mosaic OFF] epoch {epoch+1}: close-mosaic phase', flush=True)

        model_head.train()
        train_loss_sum   = 0.0;  train_count = 0
        coco_batch_count = 0;    neubie_batch_count = 0
        epoch_cls_counts = Counter()
        t0 = time.time()

        # ── Build per-epoch train iterator ────────────────────────────────────
        if stage == 1:
            _loader    = coco_train_loader
            train_iter = ((b, 'coco') for b in _loader)
            n_steps    = len(_loader)
        elif stage in (2, 4):
            _loader    = neubie_train_loader
            train_iter = ((b, 'neubie') for b in _loader)
            n_steps    = len(_loader)
        else:  # stage 3
            coco_ratio = get_coco_ratio(epoch, cfg)
            train_iter = MixedBatchIterator(coco_train_loader, neubie_train_loader,
                                            coco_ratio=coco_ratio)
            n_steps    = len(train_iter)

        for batch_idx, (batch, dataset_tag) in enumerate(
            tqdm(train_iter, total=n_steps,
                 desc=f'Train E{epoch+1}/{NUM_EPOCHS} [S{stage}]',
                 dynamic_ncols=False, ncols=120,
                 file=sys.stdout if sys.stdout.isatty() else None)
        ):
            images, boxes, labels = batch
            images = images.to(device, dtype=torch.float, non_blocking=True)
            boxes  = [b.to(device, dtype=torch.float,  non_blocking=True) for b in boxes]
            labels = [l.to(device, dtype=torch.int,    non_blocking=True) for l in labels]

            if dataset_tag == 'neubie':
                neubie_batch_count += 1
                for l_ in labels:
                    epoch_cls_counts.update(l_.cpu().tolist())
            else:
                coco_batch_count += 1

            # Select class weights for both COCO and Neubie batches
            cls_w = neubie_class_weights if dataset_tag == 'neubie' else coco_class_weights

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_amp):
                feats   = dino_backbone(images)
                outputs = model_head(feats, dataset=dataset_tag)
                strides = compute_strides(images, outputs)
                loss    = compute_loss(
                    outputs, boxes, labels, images.shape[2:], strides,
                    FOCAL_ALPHA, FOCAL_GAMMA, WEIGHT_REG, WEIGHT_CTR,
                    weight_angle=WEIGHT_ANGLE,
                    tal_topk=TAL_TOPK, tal_alpha=TAL_ALPHA, tal_beta=TAL_BETA,
                    epoch=epoch, num_epochs=NUM_EPOCHS,
                    prog_loss_epochs=PROG_LOSS_EPOCHS,
                    class_weights=cls_w,
                    vfl_alpha=VFL_ALPHA,
                    vfl_cw_neg=VFL_CW_NEG,
                    vfl_q_floor=VFL_Q_FLOOR,
                    reg_max=DFL_REG_MAX,
                    use_centerness=USE_CENTERNESS,
                    aux_loss_weight=AUX_LOSS_WEIGHT if dataset_tag == 'neubie' else 0.0,
                )

            scaler.scale(loss[0]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model_head)
            if ema_backbone is not None:
                ema_backbone.update(dino_backbone)

            train_loss_sum += float(loss[0].item())
            train_count    += 1
            global_step    += 1

            if SAVE_MODEL and args.save_every_steps > 0 and \
               global_step % args.save_every_steps == 0:
                save_checkpoint(
                    path=os.path.join(run_dir, 'last.pth'), epoch=epoch,
                    model_head=model_head, dino_backbone=dino_backbone,
                    optimizer=optimizer, cfg_dict=cfg_dict, best_val_loss=best_val_loss,
                    scaler_state=scaler.state_dict() if use_amp else None,
                    ema_state=ema.state_dict() if ema else None,
                    ema_full_state=ema.full_state() if ema else None,
                )

        train_loss_avg = train_loss_sum / max(1, train_count)
        train_time     = time.time() - t0

        # ── Validation on Neubie val set (Neubie cls only) ────────────────────
        eval_head = ema.ema if ema is not None else model_head
        # Phase 3: validate with the EMA backbone too (else EMA-head + live-backbone
        # mismatch). Phase 2: backbone is frozen, so the live backbone == EMA backbone.
        eval_backbone = ema_backbone.ema if ema_backbone is not None else dino_backbone
        eval_head.eval()
        eval_backbone.eval()
        model_head.eval()

        # Compute expensive per-image decode+NMS+F1 every VAL_METRIC_EVERY epochs.
        # All other epochs run only the fast GPU forward+loss pass (~1-2 min vs ~40 min
        # over this 33k-image val set with OBB rotated-NMS).
        # NOTE: the old `stage >= 3` clause forced full metrics EVERY epoch — with the
        # warm-start schedule (all epochs are stage 3/4) that meant ~40 min/epoch of
        # val. Removed so VAL_METRIC_EVERY is actually honored. The last epoch always
        # computes so the final checkpoint is selected on real F1.
        compute_metrics = (
            VAL_METRIC_EVERY <= 1 or
            (epoch + 1) % VAL_METRIC_EVERY == 0 or
            epoch == NUM_EPOCHS - 1
        )

        val_loss_sum = 0.0;  val_cls_sum = 0.0;  val_reg_sum = 0.0
        val_ctr_sum  = 0.0;  val_ang_sum = 0.0;  val_count   = 0
        class_stats  = {c: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0,
                             'score_sum': 0.0, 'score_count': 0}
                        for c in range(NUM_NEUBIE_CLASSES)}
        per_class_thresh = [VAL_RARE_THRESH if name in rare_classes else VAL_SCORE_THRESH
                            for name in neubie_train.class_names]

        with torch.no_grad():
            for images, boxes, labels in tqdm(neubie_val_loader,
                                              desc=f'Val E{epoch+1} [neubie]',
                                              dynamic_ncols=False, ncols=120,
                                              file=sys.stdout if sys.stdout.isatty() else None):
                images = images.to(device, dtype=torch.float, non_blocking=True)
                boxes  = [b.to(device, dtype=torch.float, non_blocking=True) for b in boxes]
                labels = [l.to(device, dtype=torch.int,   non_blocking=True) for l in labels]

                with autocast(device_type='cuda', enabled=use_amp):
                    feats   = eval_backbone(images)
                    # Evaluate the SAME (EMA) weights that get saved as best.pth —
                    # both the reported val loss and the detection metrics below
                    # must come from eval_head, or selection and saved weights
                    # disagree. One forward pass serves both.
                    eval_outputs = eval_head(feats, dataset='neubie')
                    strides = compute_strides(images, eval_outputs)
                    loss    = compute_loss(
                        eval_outputs, boxes, labels, images.shape[2:], strides,
                        FOCAL_ALPHA, FOCAL_GAMMA, WEIGHT_REG, WEIGHT_CTR,
                        weight_angle=WEIGHT_ANGLE,
                        tal_topk=TAL_TOPK, tal_alpha=TAL_ALPHA, tal_beta=TAL_BETA,
                        epoch=epoch, num_epochs=NUM_EPOCHS,
                        prog_loss_epochs=PROG_LOSS_EPOCHS,
                        class_weights=neubie_class_weights,
                        vfl_alpha=VFL_ALPHA,
                        vfl_cw_neg=VFL_CW_NEG,
                        vfl_q_floor=VFL_Q_FLOOR,
                        reg_max=DFL_REG_MAX,
                        use_centerness=USE_CENTERNESS,
                        aux_loss_weight=0.0,  # no aux loss during validation
                    )

                val_loss_sum += float(loss[0]); val_cls_sum += float(loss[1])
                val_reg_sum  += float(loss[2]); val_ctr_sum += float(loss[3])
                val_ang_sum  += float(loss[4]); val_count   += 1

                if compute_metrics:
                    img_h, img_w = images.shape[-2], images.shape[-1]
                    for bi in range(images.shape[0]):
                        one_out = {k: [o[bi:bi+1] for o in v]
                                   for k, v in eval_outputs.items()}
                        pred_xywhr, pred_scores, pred_labels_t = decode_outputs_OBB(
                            one_out, (img_h, img_w), strides=strides,
                            score_thresh=per_class_thresh,
                            nms_thresh=VAL_NMS_THRESH,
                            max_detections=300,
                            use_centerness=USE_CENTERNESS,
                        )
                        pred_xyxy    = xywhr2xyxy_aabb(pred_xywhr)
                        gt_boxes_abs = abs_gt_boxes(boxes[bi], img_h, img_w)
                        gt_labs      = labels[bi].long()
                        update_det_metrics(class_stats, pred_xyxy, pred_scores,
                                           pred_labels_t.long(), gt_boxes_abs, gt_labs,
                                           iou_thresh=VAL_IOU_THRESH)

        val_loss_avg = val_loss_sum / max(1, val_count)
        val_cls_avg  = val_cls_sum  / max(1, val_count)
        val_reg_avg  = val_reg_sum  / max(1, val_count)
        val_ctr_avg  = val_ctr_sum  / max(1, val_count)
        val_ang_avg  = val_ang_sum  / max(1, val_count)

        # ── Macro-F1 — only valid on metric epochs (per-image decode was run) ──
        val_crucial_f1: Optional[float] = None
        if compute_metrics:
            f1_per_class = []
            crucial_f1 = []
            for c in range(NUM_NEUBIE_CLASSES):
                st = class_stats[c]
                if st['gt'] == 0:
                    continue
                prec_c = st['tp'] / max(1, st['tp'] + st['fp'])
                rec_c  = st['tp'] / max(1, st['tp'] + st['fn'])
                f1c = 2 * prec_c * rec_c / max(1e-9, prec_c + rec_c)
                f1_per_class.append(f1c)
                if neubie_train.class_names[c] in crucial_small:
                    crucial_f1.append(f1c)
            val_macro_f1: Optional[float] = (
                float(sum(f1_per_class) / len(f1_per_class)) if f1_per_class else 0.0)
            val_crucial_f1 = (
                float(sum(crucial_f1) / len(crucial_f1)) if crucial_f1 else None)
        else:
            val_macro_f1 = None  # not computed this epoch

        f1_str = f'{val_macro_f1:.4f}' if val_macro_f1 is not None else 'skip'
        print(
            f'Epoch {epoch+1}/{NUM_EPOCHS} [S{stage}] | '
            f'train={train_loss_avg:.6f} | '
            f'val_neubie={val_loss_avg:.6f} '
            f'(cls={val_cls_avg:.4f} reg={val_reg_avg:.4f} '
            f'ctr={val_ctr_avg:.4f} ang={val_ang_avg:.4f}) | '
            f'macroF1={f1_str} | '
            f'COCO={coco_batch_count} Neubie={neubie_batch_count} batches | '
            f'time={train_time:.1f}s',
            flush=True,
        )
        if compute_metrics:
            cf1_str = f'{val_crucial_f1:.4f}' if val_crucial_f1 is not None else 'n/a'
            print(f'  [Crucial-small macroF1 @IoU{VAL_IOU_THRESH}] {cf1_str} '
                  f'(bollard/scooter/warning_light/traffic_light_*)', flush=True)
            print(f'  [Focus Val @IoU{VAL_IOU_THRESH}] '
                  f'{summarize_metrics(class_stats, neubie_train.class_names, focus_names)}',
                  flush=True)

        # ── Checkpointing ─────────────────────────────────────────────────────
        if SAVE_MODEL:
            ema_state      = ema.state_dict() if ema else None
            ema_full_state = ema.full_state()  if ema else None
            scaler_state   = scaler.state_dict() if use_amp else None

            save_checkpoint(
                path=os.path.join(run_dir, 'last.pth'), epoch=epoch,
                model_head=model_head, dino_backbone=dino_backbone,
                optimizer=optimizer, cfg_dict=cfg_dict, best_val_loss=best_val_loss,
                scaler_state=scaler_state, ema_state=ema_state,
                ema_full_state=ema_full_state,
                save_backbone=(UNFREEZE_LAST_N > 0),   # Phase 3: persist trained backbone
            )
            # Phase 3: also persist the EMA backbone (validation/inference uses it)
            if ema_backbone is not None:
                torch.save(ema_backbone.state_dict(),
                           os.path.join(run_dir, 'last_backbone.pth'))

            # Per-epoch snapshot (EMA weights) every N epochs
            save_every = args.save_every_epochs
            if save_every > 0 and (epoch + 1) % save_every == 0:
                snap_sd = ema_state if ema else model_head.state_dict()
                torch.save({'epoch': epoch, 'model_head': snap_sd, 'stage': stage},
                           os.path.join(run_dir, f'model_e{epoch+1:03d}.pth'))
                print(f'  [CKPT] saved model_e{epoch+1:03d}.pth', flush=True)

            # Only track best / patience from stage 2 onwards
            # (stage 1 trains only COCO cls — Neubie val metrics are meaningless)
            # best_val_loss holds the best selection SCORE for the active metric:
            #   BEST_METRIC='f1'   → macro-F1, higher is better
            #   BEST_METRIC='loss' → val loss,  lower  is better
            # On non-metric epochs (val_macro_f1 is None), patience still ticks
            # so early stopping is not artificially delayed; only best checkpoint
            # selection is skipped (we refuse to claim a "best" without F1).
            past_warmup = (epoch >= cfg.STAGE2_END + PROG_LOSS_EPOCHS - 1)
            if past_warmup:
                if BEST_METRIC == 'f1':
                    if val_macro_f1 is not None:
                        cur_score   = val_macro_f1
                        is_new_best = (best_val_loss is None or cur_score > best_val_loss)
                        if is_new_best:
                            best_val_loss    = cur_score
                            patience_counter = 0
                            best_sd  = ema_state if ema else model_head.state_dict()
                            best_ckpt = {
                                'epoch':         epoch,
                                'model_head':    best_sd,
                                'cfg':           cfg_dict,
                                'best_val_loss': best_val_loss,
                                'best_metric':   BEST_METRIC,
                            }
                            torch.save(best_ckpt, os.path.join(run_dir, 'best.pth'))
                            torch.save(best_sd, os.path.join(run_dir, f'best_e{epoch+1:03d}.pth'))
                            if ema_backbone is not None:
                                torch.save(ema_backbone.state_dict(),
                                           os.path.join(run_dir, f'best_backbone_e{epoch+1:03d}.pth'))
                                torch.save(ema_backbone.state_dict(),
                                           os.path.join(run_dir, 'best_backbone.pth'))
                            print(f'  [CKPT] ★ NEW BEST (EMA neubie {BEST_METRIC}) = '
                                  f'{best_val_loss:.6f}  → best_e{epoch+1:03d}.pth', flush=True)
                        else:
                            patience_counter += 1
                    else:
                        patience_counter += 1  # non-metric epoch: patience ticks normally
                else:
                    cur_score   = val_loss_avg
                    is_new_best = (best_val_loss is None or cur_score < best_val_loss)
                    if is_new_best:
                        best_val_loss    = cur_score
                        patience_counter = 0
                        best_sd  = ema_state if ema else model_head.state_dict()
                        best_ckpt = {
                            'epoch':         epoch,
                            'model_head':    best_sd,
                            'cfg':           cfg_dict,
                            'best_val_loss': best_val_loss,
                            'best_metric':   BEST_METRIC,
                        }
                        torch.save(best_ckpt, os.path.join(run_dir, 'best.pth'))
                        torch.save(best_sd, os.path.join(run_dir, f'best_e{epoch+1:03d}.pth'))
                        if ema_backbone is not None:
                            torch.save(ema_backbone.state_dict(),
                                       os.path.join(run_dir, f'best_backbone_e{epoch+1:03d}.pth'))
                            torch.save(ema_backbone.state_dict(),
                                       os.path.join(run_dir, 'best_backbone.pth'))
                        print(f'  [CKPT] ★ NEW BEST (EMA neubie {BEST_METRIC}) = '
                              f'{best_val_loss:.6f}  → best_e{epoch+1:03d}.pth', flush=True)
                    else:
                        patience_counter += 1

            rec = {
                'epoch': epoch, 'stage': stage, 'global_step': global_step,
                'lr_shared': current_lrs['shared'],
                'lr_cls_coco': current_lrs['cls_coco'],
                'lr_cls_neubie': current_lrs['cls_neubie'],
                'train_loss': train_loss_avg,
                'val_loss_neubie': val_loss_avg,
                'val_cls': val_cls_avg, 'val_reg': val_reg_avg,
                'val_ctr': val_ctr_avg, 'val_ang': val_ang_avg,
                'val_macro_f1': val_macro_f1,   # None on non-metric epochs
                'val_crucial_f1': val_crucial_f1,  # macro-F1 over crucial small classes
                'metrics_computed': compute_metrics,
                'best_metric': BEST_METRIC,
                'coco_batches': coco_batch_count,
                'neubie_batches': neubie_batch_count,
                'mosaic_on': use_mosaic_this_epoch,
                'ema_decay': ema._d() if ema else None,
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

        past_warmup_es = (epoch >= cfg.STAGE2_END + PROG_LOSS_EPOCHS - 1)
        if past_warmup_es and args.patience > 0 and patience_counter >= args.patience:
            print(f'Early stopping (patience={args.patience}).', flush=True)
            break

    print('Training complete.', flush=True)


if __name__ == '__main__':
    main()
