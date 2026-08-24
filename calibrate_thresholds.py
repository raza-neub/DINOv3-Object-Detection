#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_thresholds.py — Per-class confidence threshold calibration (Priority 4)
==================================================================================

For each class c, finds the threshold that maximises F1 on the validation set:

    tau_c = argmax_tau F1_c(tau)

Supports both mix (DINODetectionHeadMix) and v3 (DINODetectionHeadV3) checkpoints.
Output: JSON file with per-class thresholds ready to be used at inference.

Run
---
  # Mix checkpoint
  python calibrate_thresholds.py --config config/config_mix.py \
      --checkpoint results/.../best.pth \
      --model-type mix --device cuda:1

  # V3 checkpoint
  python calibrate_thresholds.py --config config/config_v3.py \
      --checkpoint results/.../best.pth \
      --model-type v3 --device cuda:0

  # Custom base threshold and output path
  python calibrate_thresholds.py --config config/config_mix.py \
      --checkpoint results/.../best.pth \
      --model-type mix \
      --base-thresh 0.01 --nms-thresh 0.6 \
      --output thresholds.json --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ── Config loader ─────────────────────────────────────────────────────────────

def load_cfg(path: str):
    spec = importlib.util.spec_from_file_location('cal_cfg', str(Path(path).resolve()))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Geometry helpers ──────────────────────────────────────────────────────────

def rel_xywh_to_abs_xyxy(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    b  = boxes.reshape(-1, 4)
    x1 = b[:, 0] * w;  y1 = b[:, 1] * h
    x2 = x1 + b[:, 2] * w;  y2 = y1 + b[:, 3] * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def xywhr_to_xyxy_norot(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.shape[0] == 0:
        return boxes.new_zeros((0, 4))
    cx = boxes[:, 0];  cy = boxes[:, 1]
    hw = boxes[:, 2] * 0.5;  hh = boxes[:, 3] * 0.5
    return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=1)


def box_iou_matrix(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    if b1.numel() == 0 or b2.numel() == 0:
        return torch.zeros(b1.shape[0], b2.shape[0])
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (a1[:, None] + a2[None, :] - inter + 1e-7)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def infer_backbone(path: str, cfg_mod):
    """Return (dino_model, dino_weights, embed_dim, n_layers)."""
    run_cfg_file = Path(path).parent / 'run_config.json'
    if run_cfg_file.exists():
        try:
            rc = json.loads(run_cfg_file.read_text())
            m, w = rc.get('DINO_MODEL', ''), rc.get('DINO_WEIGHTS', '')
            if m and w:
                ed = cfg_mod.MODEL_TO_EMBED_DIM.get(m, 384)
                nl = cfg_mod.MODEL_TO_NUM_LAYERS.get(m, 12)
                return m, w, int(ed), int(nl)
        except Exception:
            pass

    try:
        ckpt = torch.load(path, map_location='cpu')
        sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
        wt   = sd.get('fpn.proj_shallow.0.weight')
        if wt is not None:
            ed  = int(wt.shape[1])
            d2m = {v: k for k, v in cfg_mod.MODEL_TO_EMBED_DIM.items()}
            m   = d2m.get(ed, cfg_mod.DINO_MODEL)
            nl  = cfg_mod.MODEL_TO_NUM_LAYERS.get(m, 12)
            return m, cfg_mod.DINO_WEIGHTS, int(ed), int(nl)
    except Exception:
        pass

    m = cfg_mod.DINO_MODEL
    return m, cfg_mod.DINO_WEIGHTS, cfg_mod.MODEL_TO_EMBED_DIM[m], cfg_mod.MODEL_TO_NUM_LAYERS[m]


def load_head_weights(path: str, model_head: torch.nn.Module):
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('ema', None)
    if sd is None:
        sd = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items()
          if not k.endswith('total_ops') and not k.endswith('total_params')}
    missing, unexpected = model_head.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [WARN] Missing  ({len(missing)}): {missing[:4]} …')
    if unexpected:
        print(f'  [WARN] Unexpected ({len(unexpected)}): {unexpected[:4]} …')


# ── Collect all raw detections (at very low threshold) ───────────────────────

def collect_detections(model_head, dino_backbone, val_loader, device,
                       base_thresh: float, nms_thresh: float,
                       dataset_tag: str, decode_fn, strides_fn,
                       use_obb: bool, decode_kwargs: dict | None = None):
    """Run inference once at base_thresh and store all (score, is_tp) per class."""
    # image_results: list of dicts with pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
    image_results = []
    extra_kw = decode_kwargs or {}

    model_head.eval()
    dino_backbone.eval()

    with torch.no_grad():
        for images, boxes, labels in tqdm(val_loader, desc='Collecting detections'):
            images = images.to(device, dtype=torch.float, non_blocking=True)
            img_h, img_w = images.shape[-2], images.shape[-1]

            feats   = dino_backbone(images)
            outputs = model_head(feats, dataset=dataset_tag) if dataset_tag else model_head(feats)
            strides = strides_fn(images, outputs)

            for bi in range(images.shape[0]):
                one_out = {k: [o[bi:bi+1] for o in v] for k, v in outputs.items()}
                if use_obb:
                    pred_xywhr, pred_scores, pred_lbls = decode_fn(
                        one_out, (img_h, img_w), strides=strides,
                        score_thresh=base_thresh,
                        nms_thresh=nms_thresh,
                        max_detections=1000,
                        **extra_kw,
                    )
                    pred_boxes = xywhr_to_xyxy_norot(pred_xywhr)
                else:
                    pred_boxes, pred_scores, pred_lbls = decode_fn(
                        one_out, (img_h, img_w), strides=strides,
                        score_thresh=base_thresh,
                        nms_thresh=nms_thresh,
                        **extra_kw,
                    )

                gt_boxes = rel_xywh_to_abs_xyxy(boxes[bi].to(device), img_h, img_w)
                gt_lbls  = labels[bi].to(device).long()

                image_results.append({
                    'pred_boxes':  pred_boxes.cpu(),
                    'pred_scores': pred_scores.cpu(),
                    'pred_labels': pred_lbls.cpu().long(),
                    'gt_boxes':    gt_boxes.cpu(),
                    'gt_labels':   gt_lbls.cpu(),
                })

    return image_results


# ── Build per-class score lists with TP/FP labels ────────────────────────────

def build_score_lists(image_results: list, num_classes: int, iou_thresh: float = 0.5):
    """Returns det_by_cls[cls] = list of (score, is_tp) and n_gt per class."""
    det_by_cls = defaultdict(list)
    n_gt       = np.zeros(num_classes, dtype=np.int64)

    for res in image_results:
        pb = res['pred_boxes'];  ps = res['pred_scores'];  pl = res['pred_labels']
        gb = res['gt_boxes'];    gl = res['gt_labels']

        for g in gl.tolist():
            if 0 <= g < num_classes:
                n_gt[g] += 1

        if pb.shape[0] == 0:
            continue

        for cls in range(num_classes):
            pm = pl == cls;  gm = gl == cls
            pb_c = pb[pm];   ps_c = ps[pm];   gb_c = gb[gm]
            if pb_c.shape[0] == 0:
                continue
            if gb_c.shape[0] == 0:
                for s in ps_c.tolist():
                    det_by_cls[cls].append((float(s), 0))
                continue

            iou_mat = box_iou_matrix(pb_c, gb_c)
            matched = torch.zeros(gb_c.shape[0], dtype=torch.bool)
            for pi in torch.argsort(ps_c, descending=True).tolist():
                best_iou, best_j = iou_mat[pi].max(0)
                best_iou = float(best_iou)
                if best_iou >= iou_thresh and not matched[best_j]:
                    matched[best_j] = True
                    det_by_cls[cls].append((float(ps_c[pi]), 1))
                else:
                    det_by_cls[cls].append((float(ps_c[pi]), 0))

    return det_by_cls, n_gt


# ── Per-class threshold search (maximise F1) ─────────────────────────────────

def find_best_threshold(scores_tp: list, n_gt: int,
                        thresh_min: float = 0.01,
                        thresh_max: float = 0.95,
                        n_steps: int = 200) -> tuple[float, float]:
    """Returns (best_threshold, best_f1).

    Sweeps `n_steps` candidate thresholds and picks the one with highest F1.
    Falls back to thresh_min if no detections are available.
    """
    if not scores_tp or n_gt == 0:
        return thresh_min, 0.0

    scores_tp_sorted = sorted(scores_tp, key=lambda x: -x[0])
    score_vals = [s for s, _ in scores_tp_sorted]
    tp_arr     = np.array([t for _, t in scores_tp_sorted], dtype=np.float32)

    thresholds = np.linspace(thresh_min, thresh_max, n_steps)
    best_f1    = -1.0
    best_tau   = thresh_min

    for tau in thresholds:
        # All detections with score >= tau
        keep = np.array(score_vals) >= tau
        n_det = int(keep.sum())
        if n_det == 0:
            # F1 = 0 when no predictions
            f1 = 0.0
        else:
            tp = int(tp_arr[keep].sum())
            fp = n_det - tp
            fn = n_gt - tp
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)

    return best_tau, best_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser('Per-class threshold calibration')
    p.add_argument('--config',      type=str, required=True)
    p.add_argument('--checkpoint',  type=str, required=True)
    p.add_argument('--model-type',  type=str, default='mix',
                   choices=['mix', 'v3', 'convnext'],
                   help='mix, v3, or convnext checkpoint')
    p.add_argument('--device',      type=str, default='')
    p.add_argument('--batch-size',  type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--base-thresh', type=float, default=0.01,
                   help='Low base threshold used when collecting all detections')
    p.add_argument('--nms-thresh',  type=float, default=0.6)
    p.add_argument('--iou-thresh',  type=float, default=0.5,
                   help='IoU threshold for TP/FP matching')
    p.add_argument('--thresh-steps', type=int, default=200,
                   help='Number of threshold candidates to sweep per class')
    p.add_argument('--output',      type=str, default='',
                   help='Output JSON path (default: <checkpoint_dir>/thresholds.json)')
    args = p.parse_args()

    device = (torch.device(args.device) if args.device
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    cfg = load_cfg(args.config)

    FPN_CH  = cfg.FPN_CH
    N_CONVS = cfg.N_CONVS
    DINOV3_DIR = cfg.DINOV3_DIR

    # ── ConvNeXt backbone uses a different wrapper — handle separately ────
    if args.model_type == 'convnext':
        from src.backbone_convnext import DinoBackboneConvNeXt
        from src.head_convnext     import ConvNeXtDetectionHeadMixP2
        from src.decode               import decode_outputs_OBB

        dino_raw = torch.hub.load(repo_or_dir=DINOV3_DIR, model=cfg.DINO_MODEL,
                                  source='local', weights=cfg.DINO_WEIGHTS)
        dino_backbone = DinoBackboneConvNeXt(dino_raw).to(device)
        for param in dino_backbone.parameters():
            param.requires_grad_(False)

        CONVNEXT_IN_CHANNELS = list(getattr(cfg, 'CONVNEXT_IN_CHANNELS', [192, 384, 768]))
        num_neubie = cfg.NUM_NEUBIE_CLASSES
        num_coco   = cfg.NUM_COCO_CLASSES

        # Phase C architectural knobs
        USE_AIFI        = bool(getattr(cfg, 'USE_AIFI', False))
        AIFI_LAYERS     = int(getattr(cfg, 'AIFI_LAYERS', 2))
        AIFI_HEADS      = int(getattr(cfg, 'AIFI_HEADS', 8))
        USE_DFL         = bool(getattr(cfg, 'USE_DFL', False))
        DFL_REG_MAX     = int(getattr(cfg, 'DFL_REG_MAX', 16)) if USE_DFL else 0
        USE_CENTERNESS  = bool(getattr(cfg, 'USE_CENTERNESS', True))
        USE_DCN         = bool(getattr(cfg, 'USE_DCN', False))
        USE_AUX_DECODER = bool(getattr(cfg, 'USE_AUX_DECODER', False))
        AUX_NUM_QUERIES = int(getattr(cfg, 'AUX_NUM_QUERIES', 100))
        AUX_DECODER_LAYERS = int(getattr(cfg, 'AUX_DECODER_LAYERS', 2))
        USE_OBB         = bool(getattr(cfg, 'USE_OBB', True))

        print(f'Model: ConvNeXtDetectionHeadMixP2  in_channels={CONVNEXT_IN_CHANNELS}  '
              f'num_neubie={num_neubie}  num_coco={num_coco}  obb={USE_OBB}')
        model_head = ConvNeXtDetectionHeadMixP2(
            in_channels_list=CONVNEXT_IN_CHANNELS,
            fpn_channels=FPN_CH,
            num_coco_classes=num_coco,
            num_neubie_classes=num_neubie,
            num_convs=N_CONVS,
            obb=USE_OBB,
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
        load_head_weights(args.checkpoint, model_head)

        DATASET_ROOT = cfg.NEUBIE_ROOT
        dataset_tag  = 'neubie'
        num_classes  = num_neubie
        decode_fn    = decode_outputs_OBB
        use_obb      = True

        from src.dataset import DatasetCOCOv3
        val_ds = DatasetCOCOv3(
            DATASET_ROOT, 'val', cfg.IMG_SIZE, cfg.PATCH_SIZE,
            augment_prob=0.0, mean=cfg.IMG_MEAN, std=cfg.IMG_STD,
            layout='custom', use_mosaic=False,
        )
        class_names = val_ds.class_names

    elif args.model_type == 'mix':
        from src.backbone_vitsplus import DinoBackboneV3
        from src.decode import decode_outputs_OBB

        dino_model, dino_weights, embed_dim, n_layers = infer_backbone(args.checkpoint, cfg)
        print(f'Backbone: {dino_model}  embed_dim={embed_dim}  n_layers={n_layers}')
        dino_raw   = torch.hub.load(repo_or_dir=DINOV3_DIR, model=dino_model,
                                    source='local', weights=dino_weights)
        dino_backbone = DinoBackboneV3(dino_raw, n_layers).to(device)
        for param in dino_backbone.parameters():
            param.requires_grad_(False)

        num_neubie = cfg.NUM_NEUBIE_CLASSES
        num_coco   = cfg.NUM_COCO_CLASSES
        is_p2   = False    # 4-level P2 (stride-8) head vs 3-level mix head
        has_obb = True     # angle branch present?
        # Infer class counts + architecture from the checkpoint
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            sd   = ckpt.get('ema', ckpt.get('model_head', ckpt))
            if isinstance(sd, dict):
                w = sd.get('head.cls_neubie.0.weight')
                if w is not None:
                    num_neubie = int(w.shape[0])
                w = sd.get('head.cls_coco.0.weight')
                if w is not None:
                    num_coco = int(w.shape[0])
                is_p2   = any(k.startswith('fpn.p2_refine') for k in sd) \
                          or 'head.cls_neubie.3.weight' in sd
                has_obb = 'head.angle_reg.0.weight' in sd
        except Exception:
            pass

        if is_p2:
            from src.head_vitsplus_mix_p2 import DINODetectionHeadMixP2 as _MixHead
        else:
            from src.head_vitsplus_mix import DINODetectionHeadMix as _MixHead
        print(f'Model: {"DINODetectionHeadMixP2 (4-level P2)" if is_p2 else "DINODetectionHeadMix (3-level)"}'
              f'  num_neubie={num_neubie}  num_coco={num_coco}  obb={has_obb}')
        model_head = _MixHead(
            backbone_out_channels=embed_dim,
            fpn_channels=FPN_CH,
            num_coco_classes=num_coco,
            num_neubie_classes=num_neubie,
            num_convs=N_CONVS,
            obb=has_obb,
        ).to(device)
        load_head_weights(args.checkpoint, model_head)

        DATASET_ROOT = cfg.NEUBIE_ROOT
        dataset_tag  = 'neubie'
        num_classes  = num_neubie
        decode_fn    = decode_outputs_OBB
        use_obb      = True

        # Load class names from dataset
        from src.dataset import DatasetCOCOv3
        val_ds = DatasetCOCOv3(
            DATASET_ROOT, 'val', cfg.IMG_SIZE, cfg.PATCH_SIZE,
            augment_prob=0.0, mean=cfg.IMG_MEAN, std=cfg.IMG_STD,
            layout='custom', use_mosaic=False,
        )
        class_names = val_ds.class_names

    else:  # v3
        from src.backbone_vitsplus import DinoBackboneV3
        from src.neck     import DINODetectionHeadV3
        from src.decode         import decode_outputs_OBB

        dino_model, dino_weights, embed_dim, n_layers = infer_backbone(args.checkpoint, cfg)
        print(f'Backbone: {dino_model}  embed_dim={embed_dim}  n_layers={n_layers}')
        dino_raw   = torch.hub.load(repo_or_dir=DINOV3_DIR, model=dino_model,
                                    source='local', weights=dino_weights)
        dino_backbone = DinoBackboneV3(dino_raw, n_layers).to(device)
        for param in dino_backbone.parameters():
            param.requires_grad_(False)

        num_classes = cfg.NUM_CLASSES
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            sd   = ckpt.get('ema', ckpt.get('model_head', ckpt))
            if isinstance(sd, dict):
                w = sd.get('head.cls_logits.0.weight')
                if w is not None:
                    num_classes = int(w.shape[0])
        except Exception:
            pass

        print(f'Model: DINODetectionHeadV3  num_classes={num_classes}')
        model_head = DINODetectionHeadV3(
            backbone_out_channels=embed_dim,
            fpn_channels=FPN_CH,
            num_classes=num_classes,
            num_convs=N_CONVS,
            obb=True,
        ).to(device)
        load_head_weights(args.checkpoint, model_head)

        DATASET_ROOT = cfg.COCO_ROOT
        dataset_tag  = None  # v3 head takes no dataset arg
        decode_fn    = decode_outputs_OBB
        use_obb      = True

        from src.dataset import DatasetCOCOv3
        val_ds = DatasetCOCOv3(
            DATASET_ROOT, 'val', cfg.IMG_SIZE, cfg.PATCH_SIZE,
            augment_prob=0.0, mean=cfg.IMG_MEAN, std=cfg.IMG_STD,
            layout='custom', use_mosaic=False,
        )
        class_names = val_ds.class_names

    # ── Data loader ───────────────────────────────────────────────────────────
    try:
        from train.collate_fn import collate_fn
    except ImportError:
        from collate_fn import collate_fn

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'Val set: {len(val_ds)} images  |  {num_classes} classes')

    # ── Strides helper ────────────────────────────────────────────────────────
    def strides_fn(images, outputs):
        img_w  = images.shape[-1]
        feat_w = outputs['cls'][0].shape[3]
        s0     = float(img_w / feat_w)
        return [s0 * (2 ** l) for l in range(len(outputs['cls']))]

    # Wrap model_head.forward to accept optional dataset arg
    class _HeadWrapper(torch.nn.Module):
        def __init__(self, head, tag):
            super().__init__()
            self.head = head
            self.tag  = tag
        def forward(self, feats, dataset=None):
            if self.tag is not None:
                return self.head(feats, dataset=self.tag)
            return self.head(feats)

    wrapped_head = _HeadWrapper(model_head, dataset_tag)

    # ── Collect detections at base_thresh ─────────────────────────────────────
    decode_kwargs = {}
    if args.model_type == 'convnext':
        decode_kwargs['use_centerness'] = USE_CENTERNESS

    print(f'\nRunning inference at base_thresh={args.base_thresh} …')
    image_results = collect_detections(
        wrapped_head, dino_backbone, val_loader, device,
        base_thresh=args.base_thresh,
        nms_thresh=args.nms_thresh,
        dataset_tag=dataset_tag,
        decode_fn=decode_fn,
        strides_fn=strides_fn,
        use_obb=use_obb,
        decode_kwargs=decode_kwargs,
    )
    print(f'Collected {len(image_results)} image results.')

    # ── Build per-class score lists ───────────────────────────────────────────
    det_by_cls, n_gt = build_score_lists(image_results, num_classes, args.iou_thresh)

    # ── Find best threshold per class ─────────────────────────────────────────
    print(f'\nSweeping {args.thresh_steps} threshold candidates per class …\n')
    thresholds = {}
    header = f'{"Class":<30}  {"n_gt":>6}  {"n_det":>6}  {"tau":>6}  {"F1":>6}'
    print(header)
    print('-' * len(header))

    for cls in range(num_classes):
        name       = class_names[cls] if cls < len(class_names) else str(cls)
        scores_tp  = det_by_cls.get(cls, [])
        best_tau, best_f1 = find_best_threshold(
            scores_tp, int(n_gt[cls]),
            thresh_min=args.base_thresh,
            thresh_max=0.95,
            n_steps=args.thresh_steps,
        )
        thresholds[name] = round(best_tau, 4)
        print(f'{name:<30}  {n_gt[cls]:>6}  {len(scores_tp):>6}  '
              f'{best_tau:>6.3f}  {best_f1:>6.3f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output or str(Path(args.checkpoint).parent / 'thresholds.json')
    result = {
        'checkpoint':    args.checkpoint,
        'model_type':    args.model_type,
        'iou_thresh':    args.iou_thresh,
        'base_thresh':   args.base_thresh,
        'nms_thresh':    args.nms_thresh,
        'num_val_images': len(image_results),
        'thresholds':    thresholds,
        # Also provide an ordered list for index-based lookup
        'thresholds_list': [thresholds.get(
            class_names[c] if c < len(class_names) else str(c), args.base_thresh)
            for c in range(num_classes)],
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved per-class thresholds → {out_path}')
    print('Use thresholds_list at inference for index-based lookup.')


if __name__ == '__main__':
    main()
