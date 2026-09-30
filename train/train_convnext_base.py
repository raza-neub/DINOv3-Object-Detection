#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_convnext.py — ConvNeXt backbone + mix_p2 detection head training
======================================================================

Copy of train_mix_p2.py adapted for ConvNeXt backbones. Changes vs train_mix_p2.py:

  - Imports DinoBackboneConvNeXt (not DinoBackboneV3)
  - Imports ConvNeXtDetectionHeadMixP2 (not DINODetectionHeadMixP2)
  - Model constructed with in_channels_list from config (not backbone_out_channels)
  - No warm-start logic (training from scratch with new backbone)
  - No backbone unfreeze support (ConvNeXt unfreeze uses stages, not blocks —
    can be added later if needed)

Everything else (training loop, loss, decode, augmentation, curriculum,
optimizer, dataset loading) is identical to train_mix_p2.py.

Run
---
    nohup python -u train/train_convnext.py \\
        --config config/config_convnext_small.py \\
        --use-amp > nohup_convnext_small.out 2>&1 &

    nohup python -u train/train_convnext.py \\
        --config config/config_convnext_base.py \\
        --use-amp --device cuda:1 > nohup_convnext_base.out 2>&1 &

Resume
------
    python train/train_convnext.py --config config/config_convnext_small.py \\
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


# ── EMA (same as train_mix_p2.py) ────────────────────────────────────────────

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
        return {'ema_weights': self.ema.state_dict(), 'updates': self.updates}

    def load_full_state(self, state: dict):
        self.ema.load_state_dict(state['ema_weights'])
        self.updates = int(state.get('updates', 0))


# ── Stage helpers ─────────────────────────────────────────────────────────────

def get_stage(epoch: int, cfg) -> int:
    if epoch < cfg.STAGE1_END:
        return 1
    elif epoch < cfg.STAGE2_END:
        return 2
    elif epoch < cfg.STAGE3_END:
        return 3
    else:
        return 4


def get_coco_ratio(epoch: int, cfg) -> float:
    if epoch < cfg.STAGE3_SUB1_END:
        return 0.6
    elif epoch < cfg.STAGE3_SUB2_END:
        return 0.4
    else:
        return 0.2


# ── RepeatFactorSampler ─────────────────────────────────────────────────────

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


# ── Mixed batch iterator (stage 3) ──────────────────────────────────────────

class MixedBatchIterator:
    def __init__(self, coco_loader: DataLoader, neubie_loader: DataLoader,
                 coco_ratio: float):
        self.coco_loader   = coco_loader
        self.neubie_loader = neubie_loader
        self.coco_ratio    = coco_ratio

    def _compute_targets(self):
        n_coco   = len(self.coco_loader)
        n_neubie = len(self.neubie_loader)
        r = max(min(self.coco_ratio, 0.999), 0.001)
        max_by_coco   = n_coco / r
        max_by_neubie = n_neubie / (1.0 - r)
        effective_total = int(min(max_by_coco, max_by_neubie))
        target_coco   = min(int(round(r * effective_total)), n_coco)
        target_neubie = min(effective_total - target_coco, n_neubie)
        return target_coco, target_neubie

    def __iter__(self) -> Iterator[Tuple[Any, str]]:
        target_coco, target_neubie = self._compute_targets()
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


# ── Utility helpers ──────────────────────────────────────────────────────────

def load_cfg_from_py(py_path: str):
    py_path = str(Path(py_path).resolve())
    spec = importlib.util.spec_from_file_location('user_cfg', py_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_run_dir(results_path, dino_model_name, run_name=None):
    ts   = _dt.datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_{dino_model_name}_convnext')
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
                    best_val_loss=None, scaler_state=None,
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
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model_head, dino_backbone, optimizer=None, ema=None,
                    map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
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
    if optimizer is not None and isinstance(ckpt, dict) and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError):
            print('  [resume] optimizer state incompatible — starting optimizer fresh')
    ema_restored = False
    if ema is not None and isinstance(ckpt, dict):
        if 'ema_full' in ckpt:
            ema.load_full_state(ckpt['ema_full'])
            ema_restored = True
        elif 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
            epoch_val = int(ckpt.get('epoch', 0))
            ema.updates = max(ema.updates, epoch_val * 500)
            ema_restored = True
    start_epoch  = int(ckpt.get('epoch', -1)) + 1 if isinstance(ckpt, dict) else 0
    best_val     = ckpt.get('best_val_loss', None) if isinstance(ckpt, dict) else None
    scaler_state = ckpt.get('scaler', None)         if isinstance(ckpt, dict) else None
    return start_epoch, best_val, scaler_state, ema_restored


# ── Validation helpers ──────────────────────────────────────────────────────

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


# ── LR helpers ──────────────────────────────────────────────────────────────

def set_lr(optimizer, group_name: str, new_lr: float):
    for pg in optimizer.param_groups:
        if pg.get('name') == group_name:
            pg['lr'] = new_lr


def adjust_lr_for_stage(optimizer, stage: int, cfg):
    if stage == 2:
        set_lr(optimizer, 'shared',     cfg.LR_SHARED_STAGE2)
        set_lr(optimizer, 'cls_coco',   0.0)
        set_lr(optimizer, 'cls_neubie', cfg.LR_CLS_NEUBIE)
    elif stage == 1:
        set_lr(optimizer, 'shared',     cfg.LR_SHARED)
        set_lr(optimizer, 'cls_coco',   cfg.LR_CLS_COCO)
        set_lr(optimizer, 'cls_neubie', 0.0)
    else:
        set_lr(optimizer, 'shared',     cfg.LR_SHARED)
        set_lr(optimizer, 'cls_coco',   cfg.LR_CLS_COCO)
        set_lr(optimizer, 'cls_neubie', cfg.LR_CLS_NEUBIE)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    p = argparse.ArgumentParser('DINOv3 ConvNeXt mixed COCO+Neubie training')
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
    p.add_argument('--compile-backbone',  action='store_true',
                   help='torch.compile the frozen backbone for ~30%% speedup')
    # Calibration fine-tuning: freeze shared+coco, train only cls_neubie
    # with VFL_Q_FLOOR=0.0 to sharpen score calibration
    p.add_argument('--calibrate',        action='store_true',
                   help='Calibration fine-tuning: freeze shared+reg, train only cls_neubie')
    p.add_argument('--calibrate-epochs', type=int, default=20,
                   help='Number of calibration fine-tuning epochs')
    p.add_argument('--calibrate-lr',     type=float, default=5e-4,
                   help='Peak LR for calibration fine-tuning')
    # Strategy 1: Fast full-head+FPN fine-tuning (10-20 epochs)
    # Unfreezes FPN + all heads (backbone stays frozen), VFL_Q_FLOOR=0.0,
    # lower LR with cosine decay, Neubie-only data, optional cls loss boost
    p.add_argument('--finetune',         action='store_true',
                   help='Fast fine-tune: unfreeze FPN+heads, VFL_Q_FLOOR=0.0, low LR')
    p.add_argument('--finetune-epochs',  type=int, default=15,
                   help='Number of fine-tuning epochs')
    p.add_argument('--finetune-lr',      type=float, default=1e-4,
                   help='Peak LR for fine-tuning (all param groups)')
    p.add_argument('--cls-loss-scale',   type=float, default=1.5,
                   help='Scale cls loss weight during fine-tuning (prioritize score fix)')
    p.add_argument('--reg-loss-scale',   type=float, default=0.5,
                   help='Scale reg loss weight during fine-tuning (deprioritize box refinement)')
    p.add_argument('--gpus',             type=str, default='',
                   help='Comma-separated GPU ids for DataParallel (e.g. "1,2")')
    p.add_argument('--trt-head',         action='store_true',
                   help='Use TRT-exportable head (GN preserved, DCN replaced with DilatedConv)')
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

    from src.dataset         import DatasetCOCOv3
    from src.backbone_convnext import DinoBackboneConvNeXt
    if args.trt_head:
        from src.head_convnext_trt import ConvNeXtDetectionHeadMixP2
        print('[TRT-HEAD] Using TRT-clean head (BN, no DCN)')
    else:
        from src.head_convnext     import ConvNeXtDetectionHeadMixP2
    from src.loss                 import compute_loss
    from src.decode               import decode_outputs_OBB
    from collate_fn                  import collate_fn

    # Multi-GPU: --gpus "1,2" uses DataParallel across those GPUs
    gpu_ids = [int(g) for g in args.gpus.split(',') if g.strip()] if args.gpus else []
    if gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f'Using DataParallel on GPUs: {gpu_ids}  (primary={device})')
    else:
        device = (torch.device(args.device) if args.device
                  else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print('Using device:', device)

    cfg_use_amp = bool(getattr(cfg, 'USE_AMP', False))
    use_amp     = (args.use_amp or cfg_use_amp) and device.type == 'cuda'
    scaler      = (GradScaler('cuda', enabled=use_amp) if device.type == 'cuda'
                   else GradScaler('cpu', enabled=False))
    print('AMP enabled:', use_amp)

    # ── Config ──────────────────────────────────────────────────────────────
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
    # In calibration/finetune mode, always compute full metrics (only ~15-20 epochs)
    if args.calibrate or args.finetune:
        VAL_METRIC_EVERY = 1

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

    # ConvNeXt-specific config
    CONVNEXT_IN_CHANNELS = list(getattr(cfg, 'CONVNEXT_IN_CHANNELS', [192, 384, 768]))

    # Phase D: SFDNet/RT-SFOD improvements
    USE_FREQ_ENHANCE  = bool(getattr(cfg, 'USE_FREQ_ENHANCE', False))
    VAR_REG_WEIGHT    = float(getattr(cfg, 'VAR_REG_WEIGHT', 0.0))
    CPD_WEIGHT        = float(getattr(cfg, 'CPD_WEIGHT', 0.0))
    CPD_WARMUP_EPOCHS = int(getattr(cfg, 'CPD_WARMUP_EPOCHS', 10))
    CPD_MOMENTUM      = float(getattr(cfg, 'CPD_MOMENTUM', 0.9))
    CPD_TEMPERATURE   = float(getattr(cfg, 'CPD_TEMPERATURE', 0.07))

    # Phase E: Head architecture overhaul
    COSINE_CLS          = bool(getattr(cfg, 'COSINE_CLS', True))
    FOCAL_WARMUP_EPOCHS = int(getattr(cfg, 'FOCAL_WARMUP_EPOCHS', 0))
    TAL_ALPHA_SCHEDULE  = bool(getattr(cfg, 'TAL_ALPHA_SCHEDULE', False))
    USE_CROSS_LEVEL_ATTN = bool(getattr(cfg, 'USE_CROSS_LEVEL_ATTN', False))

    def _worker_init_fn(worker_id):
        """Seed each worker differently so augmentations vary across workers."""
        import numpy as np
        seed = torch.initial_seed() % (2**32) + worker_id
        np.random.seed(seed)
        random.seed(seed)

    n_workers_kw = dict(
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )

    # Small-object augmentation knobs
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

    # ── Neubie dataset ──────────────────────────────────────────────────────
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

    print('Building Neubie class counts ...')
    neubie_counts, neubie_img_cls_map = get_class_counts_fast(neubie_train)
    neubie_sampler = RepeatFactorSampler(
        neubie_img_cls_map, neubie_counts, len(neubie_train))
    print(f'  Neubie RepeatFactorSampler: {len(neubie_sampler)} samples/epoch '
          f'(dataset: {len(neubie_train)})')

    neubie_class_weights = make_effective_number_weights(
        neubie_counts, len(neubie_train.class_names)).to(device)

    neubie_train_loader = DataLoader(
        neubie_train, batch_size=BATCH_SIZE, sampler=neubie_sampler,
        shuffle=False, drop_last=True, collate_fn=collate_fn, **n_workers_kw)
    neubie_val_loader = DataLoader(
        neubie_val, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, **n_workers_kw)

    # ── COCO dataset ────────────────────────────────────────────────────────
    coco_train = DatasetCOCOv3(
        COCO_ROOT_MIX, 'train', IMG_SIZE, PATCH_SIZE,
        augment_prob=PROB_AUG_TRAIN, mean=IMG_MEAN, std=IMG_STD,
        layout='coco', use_mosaic=USE_MOSAIC, **aug_kw,
    )

    assert len(coco_train.class_names) == NUM_COCO_CLASSES, (
        f'Expected {NUM_COCO_CLASSES} COCO classes, got {len(coco_train.class_names)}')

    print('Building COCO class counts ...')
    coco_counts, coco_img_cls_map = get_class_counts_fast(coco_train)
    coco_sampler = RepeatFactorSampler(
        coco_img_cls_map, coco_counts, len(coco_train))
    print(f'  COCO RepeatFactorSampler: {len(coco_sampler)} samples/epoch '
          f'(dataset: {len(coco_train)})')

    coco_class_weights = make_effective_number_weights(
        coco_counts, len(coco_train.class_names)).to(device)

    coco_train_loader = DataLoader(
        coco_train, batch_size=BATCH_SIZE, sampler=coco_sampler,
        shuffle=False, drop_last=True, collate_fn=collate_fn, **n_workers_kw)

    print(f'Neubie classes ({NUM_NEUBIE_CLASSES}): '
          f'{neubie_train.class_names}')
    print(f'COCO classes  ({NUM_COCO_CLASSES}): '
          f'{coco_train.class_names[:10]} ... (first 10)')

    # ── Model (ConvNeXt backbone + ConvNeXt detection head) ──────────────────
    dino_model    = torch.hub.load(repo_or_dir=DINOV3_DIR, model=DINO_MODEL,
                                   source='local', weights=DINO_WEIGHTS)
    dino_backbone = DinoBackboneConvNeXt(dino_model).to(device)

    actual_neubie_classes = len(neubie_train.class_names)
    actual_coco_classes   = len(coco_train.class_names)
    assert actual_neubie_classes == NUM_NEUBIE_CLASSES, (
        f'Config NUM_NEUBIE_CLASSES={NUM_NEUBIE_CLASSES} but dataset has '
        f'{actual_neubie_classes} classes.')
    assert actual_coco_classes == NUM_COCO_CLASSES, (
        f'Config NUM_COCO_CLASSES={NUM_COCO_CLASSES} but dataset has '
        f'{actual_coco_classes} classes.')

    model_head = ConvNeXtDetectionHeadMixP2(
        in_channels_list=CONVNEXT_IN_CHANNELS,
        fpn_channels=FPN_CH,
        num_coco_classes=actual_coco_classes,
        num_neubie_classes=actual_neubie_classes,
        num_convs=N_CONVS,
        obb=USE_OBB,
        cosine_cls=COSINE_CLS,
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
        # Phase D
        use_freq_enhance=USE_FREQ_ENHANCE,
        # Phase E
        use_cross_level_attn=USE_CROSS_LEVEL_ATTN,
    ).to(device)
    print(f'ConvNeXt detection head: 4-level pyramid [P2,P3,P4,P5]  '
          f'OBB={"ENABLED" if USE_OBB else "DISABLED"}')
    print(f'  backbone={DINO_MODEL}  in_channels={CONVNEXT_IN_CHANNELS}')
    print(f'  AIFI={USE_AIFI} DFL={USE_DFL}(reg_max={DFL_REG_MAX}) '
          f'CTR={USE_CENTERNESS} DCN={USE_DCN} AUX={USE_AUX_DECODER}')
    print(f'  [Phase D] FreqEnhance={USE_FREQ_ENHANCE} VarReg={VAR_REG_WEIGHT} '
          f'CPD={CPD_WEIGHT}(warmup={CPD_WARMUP_EPOCHS})')
    print(f'  [Phase E] CosineClassifier={COSINE_CLS} '
          f'FocalWarmup={FOCAL_WARMUP_EPOCHS}ep TAL_ALPHA_schedule={TAL_ALPHA_SCHEDULE} '
          f'CrossLevelAttn={USE_CROSS_LEVEL_ATTN}')

    # ── Backbone freeze (ConvNeXt always frozen for now) ────────────────────
    for p_ in dino_backbone.parameters():
        p_.requires_grad = False
    dino_backbone.eval()
    print('  [backbone] ConvNeXt fully frozen')

    # ── torch.compile for frozen backbone (inference-only, safe to compile) ──
    if args.compile_backbone:
        # 'default' mode: torch.inductor, minimal memory overhead, ~15-20% speedup.
        # ('reduce-overhead' uses CUDA graphs which add ~11GB GPU memory — too much
        #  when batch_size already fills the GPU.)
        print('  [backbone] Compiling with torch.compile (default) ...')
        dino_backbone = torch.compile(dino_backbone, mode='default')
        # Warm-up: one dummy forward to trigger compilation before training
        with torch.no_grad():
            _dummy = torch.randn(1, 3, *IMG_SIZE, device=device)
            _ = dino_backbone(_dummy)
            del _dummy
        print('  [backbone] Compilation done.')

    total_params = sum(p_.numel() for p_ in model_head.parameters())
    shared_count = sum(p_.numel() for p_ in model_head.shared_params())
    print(f'Head params: {total_params/1e6:.2f} M  '
          f'(shared={shared_count/1e6:.2f} M, '
          f'cls_coco={sum(p_.numel() for p_ in model_head.coco_cls_params())/1e6:.2f} M, '
          f'cls_neubie={sum(p_.numel() for p_ in model_head.neubie_cls_params())/1e6:.2f} M)')

    # ── DataParallel wrapping ──────────────────────────────────────────────
    # Keep raw references for param groups / EMA / checkpoint;
    # use _dp_* wrappers in forward passes only.
    model_head_raw    = model_head
    dino_backbone_raw = dino_backbone
    if len(gpu_ids) > 1:
        dino_backbone = torch.nn.DataParallel(dino_backbone, device_ids=gpu_ids)
        model_head    = torch.nn.DataParallel(model_head,    device_ids=gpu_ids)
        print(f'  [DataParallel] backbone + head wrapped on GPUs {gpu_ids}')

    # ── CPD prototype bank (Phase D) ───────────────────────────────────────
    cpd_bank = None
    if CPD_WEIGHT > 0:
        from src.loss import CPDPrototypeBank
        # cls_channels = 256 (same as FPN_CH for the cls tower)
        cpd_feat_dim = FPN_CH  # cls tower output dim
        cpd_bank = CPDPrototypeBank(
            num_classes=actual_neubie_classes,
            feat_dim=cpd_feat_dim,
            momentum=CPD_MOMENTUM,
            temperature=CPD_TEMPERATURE,
        ).to(device)
        print(f'CPD prototype bank: {actual_neubie_classes} classes, '
              f'dim={cpd_feat_dim}, momentum={CPD_MOMENTUM}')

    # ── EMA ─────────────────────────────────────────────────────────────────
    ema: Optional[ModelEMA] = None
    if USE_EMA:
        ema = ModelEMA(model_head_raw, decay=EMA_DECAY, tau=EMA_TAU)
        print(f'EMA enabled (head): decay={EMA_DECAY}, tau={EMA_TAU}')

    # ── Calibration / Fine-tuning mode setup ─────────────────────────────────
    CALIBRATE = args.calibrate
    FINETUNE  = args.finetune
    if CALIBRATE and FINETUNE:
        raise ValueError('--calibrate and --finetune are mutually exclusive')

    if CALIBRATE:
        if not args.resume:
            raise ValueError('--calibrate requires --resume to load a pre-trained checkpoint')
        print('\n' + '='*68)
        print('  CALIBRATION MODE: freeze shared+coco, train only cls_neubie')
        print(f'  VFL_Q_FLOOR=0.0  calibrate_lr={args.calibrate_lr}  '
              f'epochs={args.calibrate_epochs}')
        print('='*68 + '\n')
        # Freeze shared and coco params
        for p_ in model_head_raw.shared_params():
            p_.requires_grad = False
        for p_ in model_head_raw.coco_cls_params():
            p_.requires_grad = False
        # Override loss settings for calibration
        VFL_Q_FLOOR     = 0.0
        WEIGHT_REG      = 0.3
        AUX_LOSS_WEIGHT = 0.0

    if FINETUNE:
        if not args.resume:
            raise ValueError('--finetune requires --resume to load a pre-trained checkpoint')
        # Override loss settings: VFL_Q_FLOOR=0.0, scale cls/reg weights
        VFL_Q_FLOOR     = 0.0
        WEIGHT_REG      = WEIGHT_REG * args.reg_loss_scale
        AUX_LOSS_WEIGHT = 0.0
        # Freeze COCO cls (not needed for Neubie deployment)
        for p_ in model_head_raw.coco_cls_params():
            p_.requires_grad = False
        ft_shared_count = sum(p_.numel() for p_ in model_head_raw.shared_params())
        ft_neubie_count = sum(p_.numel() for p_ in model_head_raw.neubie_cls_params())
        print('\n' + '='*68)
        print('  FINETUNE MODE: unfreeze FPN + shared towers + cls_neubie')
        print(f'  VFL_Q_FLOOR=0.0  finetune_lr={args.finetune_lr}  '
              f'epochs={args.finetune_epochs}')
        print(f'  cls_loss_scale={args.cls_loss_scale}  '
              f'reg_loss_scale={args.reg_loss_scale}  '
              f'WEIGHT_REG={WEIGHT_REG:.2f}')
        print(f'  Trainable: shared+FPN={ft_shared_count/1e6:.2f}M  '
              f'cls_neubie={ft_neubie_count/1e6:.2f}M  '
              f'(coco cls FROZEN)')
        print('='*68 + '\n')

    # ── AdamW with per-component LR ─────────────────────────────────────────
    CLS_WEIGHT_DECAY = float(getattr(cfg, 'CLS_WEIGHT_DECAY', 1e-5))
    GRAD_CLIP_NORM   = float(getattr(cfg, 'GRAD_CLIP_NORM', 5.0))

    if CALIBRATE:
        opt_groups = [
            {'params': [p_ for p_ in model_head_raw.neubie_cls_params() if p_.requires_grad],
             'lr': args.calibrate_lr,
             'name': 'cls_neubie', 'weight_decay': CLS_WEIGHT_DECAY},
        ]
        optimizer = optim.AdamW(opt_groups)
        print(f'AdamW (calibrate): LR_cls_neubie={args.calibrate_lr}, '
              f'CLS_WD={CLS_WEIGHT_DECAY}')
    elif FINETUNE:
        opt_groups = [
            {'params': [p_ for p_ in model_head_raw.shared_params() if p_.requires_grad],
             'lr': args.finetune_lr,
             'name': 'shared',     'weight_decay': WEIGHT_DECAY},
            {'params': [p_ for p_ in model_head_raw.neubie_cls_params() if p_.requires_grad],
             'lr': args.finetune_lr,
             'name': 'cls_neubie', 'weight_decay': CLS_WEIGHT_DECAY},
        ]
        optimizer = optim.AdamW(opt_groups)
        print(f'AdamW (finetune): LR={args.finetune_lr}, WD={WEIGHT_DECAY}, '
              f'CLS_WD={CLS_WEIGHT_DECAY}')
    else:
        opt_groups = [
            {'params': model_head_raw.shared_params(),     'lr': cfg.LR_SHARED,
             'name': 'shared',     'weight_decay': WEIGHT_DECAY},
            {'params': model_head_raw.coco_cls_params(),   'lr': cfg.LR_CLS_COCO,
             'name': 'cls_coco',   'weight_decay': CLS_WEIGHT_DECAY},
            {'params': model_head_raw.neubie_cls_params(), 'lr': cfg.LR_CLS_NEUBIE,
             'name': 'cls_neubie', 'weight_decay': CLS_WEIGHT_DECAY},
        ]
        optimizer = optim.AdamW(opt_groups)
        print(f'AdamW: LR_shared={cfg.LR_SHARED}, LR_cls={cfg.LR_CLS_NEUBIE}, '
              f'CLS_WD={CLS_WEIGHT_DECAY}')

    # Stage peak LRs
    STAGE_PEAK_LR = {
        'shared':     {1: cfg.LR_SHARED, 2: cfg.LR_SHARED_STAGE2,
                       3: cfg.LR_SHARED, 4: cfg.LR_SHARED},
        'cls_coco':   {1: cfg.LR_CLS_COCO, 2: 0.0,
                       3: cfg.LR_CLS_COCO, 4: cfg.LR_CLS_COCO},
        'cls_neubie': {1: 0.0, 2: cfg.LR_CLS_NEUBIE,
                       3: cfg.LR_CLS_NEUBIE, 4: cfg.LR_CLS_NEUBIE},
    }
    STAGE_START = {1: 0, 2: cfg.STAGE1_END, 3: cfg.STAGE2_END, 4: cfg.STAGE3_END}
    STAGE_LEN   = {1: cfg.STAGE1_END,
                   2: cfg.STAGE2_END   - cfg.STAGE1_END,
                   3: cfg.STAGE3_END   - cfg.STAGE2_END,
                   4: NUM_EPOCHS       - cfg.STAGE3_END}

    def compute_lr(group_name: str, stage: int, epoch_in_stage: int) -> float:
        peak = STAGE_PEAK_LR[group_name][stage]
        if peak == 0.0:
            return 0.0
        stage_len = STAGE_LEN[stage]
        warmup = min(LR_WARMUP_EPOCHS, max(1, (stage_len + 1) // 2))
        if epoch_in_stage < warmup:
            return LR_MIN + (peak - LR_MIN) * (epoch_in_stage + 1) / warmup
        progress = (epoch_in_stage - warmup) / max(1, stage_len - warmup)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN + (peak - LR_MIN) * cosine

    # ── Run dir ─────────────────────────────────────────────────────────────
    run_dir  = build_run_dir(RESULTS_PATH, DINO_MODEL, args.run_name or None) if SAVE_MODEL else ''
    cfg_dict = collect_cfg_dict(args, cfg)
    if SAVE_MODEL:
        with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)

    start_epoch:  int   = 0
    best_val_loss: Optional[float] = None
    if args.resume:
        # In calibration/finetune mode, load weights only — optimizer is fresh (different param groups)
        resume_opt = None if (CALIBRATE or FINETUNE) else optimizer
        start_epoch, best_val_loss, scaler_state, ema_restored = load_checkpoint(
            args.resume, model_head=model_head_raw, dino_backbone=dino_backbone_raw,
            optimizer=resume_opt, ema=ema, map_location='cpu',
        )
        if CALIBRATE or FINETUNE:
            # Reset best_val_loss so finetuned/calibrated model competes from scratch
            best_val_loss = None
        if use_amp and scaler_state is not None:
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as e:
                print(f'[WARN] scaler state load failed: {e}')
        if ema is not None:
            if ema_restored:
                print(f'  EMA restored: updates={ema.updates}, decay={ema._d():.6f}')
            else:
                print(f'  [WARN] EMA state not found in checkpoint -- using fresh EMA')
        print(f'Resumed from {args.resume} @ epoch={start_epoch}')

    # ── Calibration / Fine-tune epoch override ──────────────────────────────
    if CALIBRATE:
        NUM_EPOCHS = start_epoch + args.calibrate_epochs
        print(f'Calibration: epochs {start_epoch+1} → {NUM_EPOCHS} '
              f'({args.calibrate_epochs} calibration epochs)')
    elif FINETUNE:
        NUM_EPOCHS = start_epoch + args.finetune_epochs
        print(f'Fine-tune: epochs {start_epoch+1} → {NUM_EPOCHS} '
              f'({args.finetune_epochs} fine-tune epochs)')

    global_step      = 0
    log_path         = os.path.join(run_dir, 'train_log.jsonl') if SAVE_MODEL else ''
    patience_counter = 0
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
    trainable_params = [p_ for p_ in model_head_raw.parameters() if p_.requires_grad]

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        stage = 4 if (CALIBRATE or FINETUNE) else get_stage(epoch, cfg)

        if stage != prev_stage:
            if FINETUNE:
                stage_desc = (f'FINETUNE: FPN+heads, VFL_Q_FLOOR=0.0, '
                              f'cls×{args.cls_loss_scale} reg×{args.reg_loss_scale}')
            elif CALIBRATE:
                stage_desc = f'CALIBRATION fine-tuning (cls_neubie only, VFL_Q_FLOOR=0.0)'
            else:
                stage_desc = {
                    1: 'COCO warmup (obj + box + COCO cls)',
                    2: 'Neubie cls warmup (Neubie only, low shared LR)',
                    3: f'Mixed COCO+Neubie (ratio={get_coco_ratio(epoch, cfg):.0%} COCO)',
                    4: 'Neubie-only calibration',
                }[stage]
            print(f'\n{"="*68}', flush=True)
            print(f'  STAGE {stage}: epoch {epoch+1}/{NUM_EPOCHS} -- {stage_desc}',
                  flush=True)
            print(f'{"="*68}\n', flush=True)
            if prev_stage >= 0:
                patience_counter = 0
                print(f'  [Patience reset at stage {prev_stage}->{stage} transition]',
                      flush=True)
            prev_stage = stage

        # ── Per-epoch LR: warmup + cosine within stage ──────────────────────
        if CALIBRATE or FINETUNE:
            _ft_epoch = epoch - start_epoch
            _ft_total = args.calibrate_epochs if CALIBRATE else args.finetune_epochs
            _ft_peak  = args.calibrate_lr if CALIBRATE else args.finetune_lr
            _ft_warmup = min(3, _ft_total // 2)
            if _ft_epoch < _ft_warmup:
                _ft_lr = LR_MIN + (_ft_peak - LR_MIN) * (_ft_epoch + 1) / _ft_warmup
            else:
                _ft_progress = (_ft_epoch - _ft_warmup) / max(1, _ft_total - _ft_warmup)
                _ft_lr = LR_MIN + (_ft_peak - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * _ft_progress))
            for pg in optimizer.param_groups:
                pg['lr'] = _ft_lr
            if CALIBRATE:
                current_lrs = {'shared': 0.0, 'cls_coco': 0.0, 'cls_neubie': _ft_lr}
                print(f'  LR (calibrate): cls_neubie={_ft_lr:.2e}', flush=True)
            else:
                current_lrs = {'shared': _ft_lr, 'cls_coco': 0.0, 'cls_neubie': _ft_lr}
                print(f'  LR (finetune): shared={_ft_lr:.2e}  cls_neubie={_ft_lr:.2e}',
                      flush=True)
        else:
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

        # ── Phase E schedule info ──────────────────────────────────────────
        if FOCAL_WARMUP_EPOCHS > 0 and not CALIBRATE and not FINETUNE:
            _cls_mode = 'FOCAL' if epoch < FOCAL_WARMUP_EPOCHS else 'VFL'
            if TAL_ALPHA_SCHEDULE:
                if epoch < FOCAL_WARMUP_EPOCHS:
                    _ep_alpha = 0.0
                elif epoch < FOCAL_WARMUP_EPOCHS + 30:
                    _ep_alpha = TAL_ALPHA * (epoch - FOCAL_WARMUP_EPOCHS) / 30.0
                else:
                    _ep_alpha = TAL_ALPHA
            else:
                _ep_alpha = TAL_ALPHA
            print(f'  [Phase E] cls_loss={_cls_mode}  TAL_ALPHA={_ep_alpha:.3f}',
                  flush=True)

        # ── Mosaic toggle ───────────────────────────────────────────────────
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

        # ── Build per-epoch train iterator ──────────────────────────────────
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

            cls_w = neubie_class_weights if dataset_tag == 'neubie' else coco_class_weights

            # CPD: always collect features for prototype update (from epoch 0),
            # but only apply contrastive loss after warmup (prototypes need to
            # stabilize first — applying loss on random prototypes hurts).
            # Also need cls_feat when VAR_REG_WEIGHT > 0 for full MARD.
            _need_cls_feat = ((CPD_WEIGHT > 0 or VAR_REG_WEIGHT > 0)
                              and dataset_tag == 'neubie')
            _cpd_w = CPD_WEIGHT if (_need_cls_feat and epoch >= CPD_WARMUP_EPOCHS) else 0.0

            # Phase E: Two-stage schedule — focal (hard targets) then VFL (soft)
            _use_focal_warmup = (FOCAL_WARMUP_EPOCHS > 0
                                 and epoch < FOCAL_WARMUP_EPOCHS
                                 and not CALIBRATE and not FINETUNE)
            # TAL_ALPHA ramp: 0.0 during focal warmup, linear ramp over next 30
            if TAL_ALPHA_SCHEDULE and not CALIBRATE and not FINETUNE:
                if epoch < FOCAL_WARMUP_EPOCHS:
                    _tal_alpha = 0.0
                elif epoch < FOCAL_WARMUP_EPOCHS + 30:
                    _tal_alpha = TAL_ALPHA * (epoch - FOCAL_WARMUP_EPOCHS) / 30.0
                else:
                    _tal_alpha = TAL_ALPHA
            else:
                _tal_alpha = TAL_ALPHA

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=use_amp):
                feats   = dino_backbone(images)
                outputs = model_head(feats, dataset=dataset_tag,
                                     return_cls_feat=_need_cls_feat)
                strides = compute_strides(images, outputs)
                loss    = compute_loss(
                    outputs, boxes, labels, images.shape[2:], strides,
                    FOCAL_ALPHA, FOCAL_GAMMA, WEIGHT_REG, WEIGHT_CTR,
                    weight_angle=WEIGHT_ANGLE,
                    tal_topk=TAL_TOPK, tal_alpha=_tal_alpha, tal_beta=TAL_BETA,
                    epoch=epoch, num_epochs=NUM_EPOCHS,
                    prog_loss_epochs=PROG_LOSS_EPOCHS,
                    class_weights=cls_w,
                    vfl_alpha=VFL_ALPHA,
                    vfl_cw_neg=VFL_CW_NEG,
                    vfl_q_floor=VFL_Q_FLOOR,
                    reg_max=DFL_REG_MAX,
                    use_centerness=USE_CENTERNESS,
                    aux_loss_weight=AUX_LOSS_WEIGHT if dataset_tag == 'neubie' else 0.0,
                    var_reg_weight=VAR_REG_WEIGHT if dataset_tag == 'neubie' else 0.0,
                    cpd_weight=_cpd_w,
                    cpd_bank=cpd_bank,
                    use_focal_warmup=_use_focal_warmup,
                )

            # In finetune mode, scale cls loss up to prioritize score recalibration
            if FINETUNE and args.cls_loss_scale != 1.0:
                total, cls_l, reg_l, ctr_l, ang_l = loss
                # Recompute total: scale cls component
                total = args.cls_loss_scale * cls_l + (total - cls_l)
                loss = (total, cls_l, reg_l, ctr_l, ang_l)

            scaler.scale(loss[0]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model_head_raw)

            train_loss_sum += float(loss[0].item())
            train_count    += 1
            global_step    += 1

            if SAVE_MODEL and args.save_every_steps > 0 and \
               global_step % args.save_every_steps == 0:
                save_checkpoint(
                    path=os.path.join(run_dir, 'last.pth'), epoch=epoch,
                    model_head=model_head_raw, dino_backbone=dino_backbone_raw,
                    optimizer=optimizer, cfg_dict=cfg_dict, best_val_loss=best_val_loss,
                    scaler_state=scaler.state_dict() if use_amp else None,
                    ema_state=ema.state_dict() if ema else None,
                    ema_full_state=ema.full_state() if ema else None,
                )

        train_loss_avg = train_loss_sum / max(1, train_count)
        train_time     = time.time() - t0

        # ── Validation on Neubie val set ────────────────────────────────────
        eval_head = ema.ema if ema is not None else model_head_raw
        eval_head.eval()
        model_head.eval()

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

        # Loss-only validation: cap at 50 batches (~5K images) for fast loss estimate.
        # Full metrics (decode + NMS + P/R/F1) only every VAL_METRIC_EVERY epochs.
        MAX_VAL_LOSS_BATCHES = 50
        n_val_batches = len(neubie_val_loader) if compute_metrics else MAX_VAL_LOSS_BATCHES

        with torch.no_grad():
            for val_bi, (images, boxes, labels) in enumerate(
                tqdm(neubie_val_loader,
                     desc=f'Val E{epoch+1} [neubie]',
                     total=n_val_batches,
                     dynamic_ncols=False, ncols=120,
                     file=sys.stdout if sys.stdout.isatty() else None)
            ):
                if not compute_metrics and val_bi >= MAX_VAL_LOSS_BATCHES:
                    break

                images = images.to(device, dtype=torch.float, non_blocking=True)
                boxes  = [b.to(device, dtype=torch.float, non_blocking=True) for b in boxes]
                labels = [l.to(device, dtype=torch.int,   non_blocking=True) for l in labels]

                with autocast(device_type='cuda', enabled=use_amp):
                    feats   = dino_backbone(images)
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
                        aux_loss_weight=0.0,
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
            val_macro_f1 = None

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

        # ── Checkpointing ───────────────────────────────────────────────────
        if SAVE_MODEL:
            ema_state      = ema.state_dict() if ema else None
            ema_full_state = ema.full_state()  if ema else None
            scaler_state   = scaler.state_dict() if use_amp else None

            save_checkpoint(
                path=os.path.join(run_dir, 'last.pth'), epoch=epoch,
                model_head=model_head_raw, dino_backbone=dino_backbone_raw,
                optimizer=optimizer, cfg_dict=cfg_dict, best_val_loss=best_val_loss,
                scaler_state=scaler_state, ema_state=ema_state,
                ema_full_state=ema_full_state,
            )

            save_every = args.save_every_epochs
            if save_every > 0 and (epoch + 1) % save_every == 0:
                snap_sd = ema_state if ema else model_head_raw.state_dict()
                torch.save({'epoch': epoch, 'model_head': snap_sd, 'stage': stage},
                           os.path.join(run_dir, f'model_e{epoch+1:03d}.pth'))
                print(f'  [CKPT] saved model_e{epoch+1:03d}.pth', flush=True)

            past_warmup = CALIBRATE or FINETUNE or (epoch >= cfg.STAGE2_END + PROG_LOSS_EPOCHS - 1)
            if past_warmup:
                if BEST_METRIC == 'f1':
                    if val_macro_f1 is not None:
                        cur_score   = val_macro_f1
                        is_new_best = (best_val_loss is None or cur_score > best_val_loss)
                        best_name = ('best_calibrated.pth' if CALIBRATE
                                     else 'best_finetuned.pth' if FINETUNE
                                     else 'best.pth')
                        if is_new_best:
                            best_val_loss    = cur_score
                            patience_counter = 0
                            best_sd  = ema_state if ema else model_head_raw.state_dict()
                            best_ckpt = {
                                'epoch':         epoch,
                                'model_head':    best_sd,
                                'cfg':           cfg_dict,
                                'best_val_loss': best_val_loss,
                                'best_metric':   BEST_METRIC,
                                'calibrated':    CALIBRATE,
                            }
                            torch.save(best_ckpt, os.path.join(run_dir, best_name))
                            torch.save(best_sd, os.path.join(run_dir, f'best_e{epoch+1:03d}.pth'))
                            label = ('CALIBRATED' if CALIBRATE
                                         else 'FINETUNED' if FINETUNE
                                         else 'EMA neubie')
                            print(f'  [CKPT] * NEW BEST ({label} {BEST_METRIC}) = '
                                  f'{best_val_loss:.6f}  -> {best_name}', flush=True)
                        else:
                            patience_counter += 1
                    else:
                        patience_counter += 1
                else:
                    cur_score   = val_loss_avg
                    is_new_best = (best_val_loss is None or cur_score < best_val_loss)
                    best_name = ('best_calibrated.pth' if CALIBRATE
                                     else 'best_finetuned.pth' if FINETUNE
                                     else 'best.pth')
                    if is_new_best:
                        best_val_loss    = cur_score
                        patience_counter = 0
                        best_sd  = ema_state if ema else model_head_raw.state_dict()
                        best_ckpt = {
                            'epoch':         epoch,
                            'model_head':    best_sd,
                            'cfg':           cfg_dict,
                            'best_val_loss': best_val_loss,
                            'best_metric':   BEST_METRIC,
                            'calibrated':    CALIBRATE,
                        }
                        torch.save(best_ckpt, os.path.join(run_dir, best_name))
                        torch.save(best_sd, os.path.join(run_dir, f'best_e{epoch+1:03d}.pth'))
                        label = ('CALIBRATED' if CALIBRATE
                                         else 'FINETUNED' if FINETUNE
                                         else 'EMA neubie')
                        print(f'  [CKPT] * NEW BEST ({label} {BEST_METRIC}) = '
                              f'{best_val_loss:.6f}  -> {best_name}', flush=True)
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
                'val_macro_f1': val_macro_f1,
                'val_crucial_f1': val_crucial_f1,
                'metrics_computed': compute_metrics,
                'best_metric': BEST_METRIC,
                'coco_batches': coco_batch_count,
                'neubie_batches': neubie_batch_count,
                'mosaic_on': use_mosaic_this_epoch,
                'ema_decay': ema._d() if ema else None,
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

        past_warmup_es = CALIBRATE or FINETUNE or (epoch >= cfg.STAGE2_END + PROG_LOSS_EPOCHS - 1)
        if past_warmup_es and args.patience > 0 and patience_counter >= args.patience:
            print(f'Early stopping (patience={args.patience}).', flush=True)
            break

    print('Training complete.', flush=True)


if __name__ == '__main__':
    main()
