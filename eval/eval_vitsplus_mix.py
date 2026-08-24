#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_mix.py — Class-by-class evaluation for DINOv3 mix model (COCO+Neubie).

Uses DinoBackboneV3 (multi-layer) + DINODetectionHeadMix.
Evaluates on the Neubie validation split (cfg.NEUBIE_ROOT) using the
Neubie classifier (dataset='neubie') — the COCO head is ignored at eval.

Metrics
───────
  Per-class   AP@50, AP@50:0.95, Precision, Recall, F1
  Global      mAP@50, mAP@50:0.95, mIoU (TP boxes)

Run
───
  python eval_mix.py --config config/config_vitsplus_mix.py \
      --checkpoint results/mixed-dataset-training/.../model_e200.pth \
      --device cuda:1

  python eval_mix.py --config config/config_vitsplus_mix.py \
      --checkpoint results/mixed-dataset-training/.../model_e200.pth \
      --score-thresh 0.05 --nms-thresh 0.6 --device cuda:1

  python eval_mix.py --config config/config_vitsplus_mix.py \
      --checkpoint results/mixed-dataset-training/.../best.pth \
      --fuse --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ── Config loader ─────────────────────────────────────────────────────────────

def load_cfg(path: str):
    spec = importlib.util.spec_from_file_location('eval_cfg', str(Path(path).resolve()))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Geometry helpers ──────────────────────────────────────────────────────────

def rel_xywh_to_abs_xyxy(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Relative XYWH → absolute XYXY."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    b = boxes.reshape(-1, 4)
    x1 = b[:, 0] * w;  y1 = b[:, 1] * h
    x2 = x1 + b[:, 2] * w;  y2 = y1 + b[:, 3] * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def xywhr_to_xyxy_norot(boxes_xywhr: torch.Tensor) -> torch.Tensor:
    """Convert OBB (cx,cy,w,h,θ) to AABB by ignoring rotation: (cx±w/2, cy±h/2).

    This is the correct comparison when GT boxes are AABB (angle=0), because
    the rotated corner hull always inflates the box, depressing IoU.
    """
    if boxes_xywhr.shape[0] == 0:
        return boxes_xywhr.new_zeros((0, 4))
    cx = boxes_xywhr[:, 0];  cy = boxes_xywhr[:, 1]
    hw = boxes_xywhr[:, 2] * 0.5;  hh = boxes_xywhr[:, 3] * 0.5
    return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=1)


def box_iou_matrix(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two XYXY box sets."""
    if b1.numel() == 0 or b2.numel() == 0:
        return torch.zeros(b1.shape[0], b2.shape[0])
    a1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    a2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    lt = torch.max(b1[:, None, :2], b2[None, :, :2])
    rb = torch.min(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(0)
    inter = wh[..., 0] * wh[..., 1]
    union = a1[:, None] + a2[None, :] - inter + 1e-7
    return inter / union


# ── AP computation ────────────────────────────────────────────────────────────

def find_best_unmatched_gt(iou_row: torch.Tensor, matched: torch.Tensor,
                            iou_thresh: float) -> tuple:
    """Return (found, gt_idx, iou) for the best unmatched GT above threshold."""
    order = torch.argsort(iou_row, descending=True)
    for gi in order.tolist():
        iou = float(iou_row[gi])
        if iou < iou_thresh:
            break
        if not matched[gi]:
            return True, gi, iou
    return False, -1, 0.0


def compute_ap_auc(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """Area under the interpolated precision-recall envelope (COCO-style AP)."""
    if n_gt == 0:
        return float('nan')
    if len(tp) == 0:
        return 0.0
    rec = tp / (n_gt + 1e-9)
    pre = tp / (tp + fp + 1e-9)
    rec = np.concatenate(([0.0], rec, [1.0]))
    pre = np.concatenate(([1.0], pre, [0.0]))
    for i in range(len(pre) - 2, -1, -1):
        pre[i] = max(pre[i], pre[i + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * pre[idx + 1]))


def match_detections(image_results: list, num_classes: int,
                     iou_thresh: float) -> tuple:
    """Match predictions to GT and compute per-class AP.

    Returns
    -------
    ap_per_class  : (num_classes,) NaN where no GT
    n_gt_per_cls  : (num_classes,)
    iou_tp_all    : list of TP IoU values
    """
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
                    det_by_cls[cls].append((float(s), 0, 0.0))
                continue

            iou_mat  = box_iou_matrix(pb_c, gb_c)
            matched  = torch.zeros(gb_c.shape[0], dtype=torch.bool)
            for pi in torch.argsort(ps_c, descending=True).tolist():
                ok, gi, iou = find_best_unmatched_gt(iou_mat[pi], matched, iou_thresh)
                if ok:
                    matched[gi] = True
                    det_by_cls[cls].append((float(ps_c[pi]), 1, iou))
                else:
                    det_by_cls[cls].append((float(ps_c[pi]), 0, 0.0))

    ap  = np.full(num_classes, float('nan'))
    iou_tps = []

    for cls in range(num_classes):
        entries = det_by_cls[cls]
        if n_gt[cls] == 0:
            continue
        if not entries:
            ap[cls] = 0.0
            continue
        entries.sort(key=lambda x: -x[0])
        tp_arr = np.cumsum([e[1] for e in entries]).astype(np.float32)
        fp_arr = np.cumsum([1 - e[1] for e in entries]).astype(np.float32)
        ap[cls] = compute_ap_auc(tp_arr, fp_arr, int(n_gt[cls]))
        iou_tps.extend(e[2] for e in entries if e[1] == 1)

    return ap, n_gt, iou_tps


def precision_recall_f1_at_thresh(image_results: list, num_classes: int,
                                   iou_thresh: float) -> tuple:
    """Compute per-class P / R / F1 at the score threshold already applied during decode."""
    tp_c = np.zeros(num_classes, dtype=np.int64)
    fp_c = np.zeros(num_classes, dtype=np.int64)
    fn_c = np.zeros(num_classes, dtype=np.int64)
    n_gt = np.zeros(num_classes, dtype=np.int64)

    for res in image_results:
        pb = res['pred_boxes'];  ps = res['pred_scores'];  pl = res['pred_labels']
        gb = res['gt_boxes'];    gl = res['gt_labels']

        for g in gl.tolist():
            if 0 <= g < num_classes:
                n_gt[g] += 1

        for cls in range(num_classes):
            pm = pl == cls;  gm = gl == cls
            pb_c = pb[pm];   ps_c = ps[pm];   gb_c = gb[gm]
            n_p = pb_c.shape[0];  n_g = gb_c.shape[0]

            if n_p == 0 and n_g == 0:
                continue
            if n_p == 0:
                fn_c[cls] += n_g
                continue
            if n_g == 0:
                fp_c[cls] += n_p
                continue

            iou_mat = box_iou_matrix(pb_c, gb_c)
            matched = torch.zeros(n_g, dtype=torch.bool)
            for pi in torch.argsort(ps_c, descending=True).tolist():
                ok, gi, _ = find_best_unmatched_gt(iou_mat[pi], matched, iou_thresh)
                if ok:
                    matched[gi] = True
                    tp_c[cls] += 1
                else:
                    fp_c[cls] += 1
            fn_c[cls] += int((~matched).sum())

    prec = tp_c / np.maximum(tp_c + fp_c, 1)
    rec  = tp_c / np.maximum(n_gt, 1)
    f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return prec, rec, f1, n_gt


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def infer_backbone_from_ckpt(path: str, cfg_mod) -> tuple[str, str, int, int]:
    """Return (dino_model, dino_weights, embed_dim, n_layers) from checkpoint.

    Priority:
      1. run_config.json saved alongside the checkpoint
      2. fpn.proj_shallow.0.weight shape in the state dict (embed_dim)
      3. Fall back to config file values
    """
    # 1. run_config.json
    run_cfg_file = Path(path).parent / 'run_config.json'
    if run_cfg_file.exists():
        try:
            run_cfg = json.loads(run_cfg_file.read_text())
            m, w = run_cfg.get('DINO_MODEL', ''), run_cfg.get('DINO_WEIGHTS', '')
            if m and w:
                ed = cfg_mod.MODEL_TO_EMBED_DIM.get(m, 384)
                nl = cfg_mod.MODEL_TO_NUM_LAYERS.get(m, 12)
                return m, w, int(ed), int(nl)
        except Exception:
            pass

    # 2. Sniff from state dict — mix model FPN key: fpn.proj_shallow.0.weight
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

    # 3. Config defaults
    m = cfg_mod.DINO_MODEL
    return m, cfg_mod.DINO_WEIGHTS, cfg_mod.MODEL_TO_EMBED_DIM[m], cfg_mod.MODEL_TO_NUM_LAYERS[m]


def infer_num_classes(path: str) -> tuple[int, int]:
    """Return (num_coco_classes, num_neubie_classes) from state dict key shapes."""
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
    num_coco   = int(sd['head.cls_coco.0.weight'].shape[0])
    num_neubie = int(sd['head.cls_neubie.0.weight'].shape[0])
    return num_coco, num_neubie


def infer_obb(path: str) -> bool:
    """Detect whether the checkpoint was trained with the OBB angle branch.

    A model trained with USE_OBB=False has no head.angle_reg.* keys. Building
    the eval head with obb=True in that case would attach a randomly-initialised
    angle branch and emit garbage angles, so we match the checkpoint instead.
    """
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
    return any(k.startswith('head.angle_reg.') for k in sd)


def load_checkpoint(path: str, model_head: torch.nn.Module):
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt
    sd   = {k: v for k, v in sd.items()
            if not k.endswith('total_ops') and not k.endswith('total_params')}
    missing, unexpected = model_head.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [WARN] Missing keys  ({len(missing)}): {missing[:4]} …')
    if unexpected:
        print(f'  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:4]} …')

    epoch = ckpt.get('epoch', None) if isinstance(ckpt, dict) else None
    if epoch is None:
        import re
        m = re.search(r'model_e?(\d+)\.pth', Path(path).name)
        epoch = int(m.group(1)) if m else '?'

    bvl = ckpt.get('best_val_loss', None) if isinstance(ckpt, dict) else None
    print(f'  Checkpoint epoch : {epoch if isinstance(epoch, str) else epoch + 1}')
    if bvl is not None:
        print(f'  Best val_loss    : {bvl:.6f}')


def collate_fn(batch):
    imgs, boxes, labels = zip(*batch)
    return torch.stack(imgs, 0), list(boxes), list(labels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser('DINOv3 mix model class-by-class evaluation')
    p.add_argument('--config',       type=str,   required=True)
    p.add_argument('--checkpoint',   type=str,   default='',
                   help='Path to model_e*.pth or best.pth')
    p.add_argument('--split',        type=str,   default='val',
                   choices=['val', 'train'])
    p.add_argument('--device',       type=str,   default='')
    p.add_argument('--batch-size',   type=int,   default=8)
    p.add_argument('--num-workers',  type=int,   default=4)
    p.add_argument('--score-thresh',    type=float, default=-1.0,
                   help='Score threshold for P/R/F1 display (overrides config VAL_SCORE_THRESH)')
    p.add_argument('--ap-score-thresh', type=float, default=0.001,
                   help='Score threshold used when collecting detections for AP (default: 0.001)')
    p.add_argument('--nms-thresh',      type=float, default=-1.0)
    p.add_argument('--iou-thresh',   type=float, default=0.5,
                   help='IoU threshold for TP matching')
    p.add_argument('--max-det',      type=int,   default=300)
    p.add_argument('--max-images',   type=int,   default=-1,
                   help='Evaluate on at most N images (default: all)')
    p.add_argument('--fuse',         action='store_true',
                   help='Fuse RepDWSBlock branches for faster inference')
    p.add_argument('--old-arch',     action='store_true',
                   help='Use reconstructed 192-ch/3×3 head (for best_e199-style ckpts)')
    p.add_argument('--layout',       type=str,   default='',
                   help='Dataset layout: custom or coco (default: from config)')
    args = p.parse_args()

    cfg = load_cfg(args.config)

    from src.dataset_eval      import DatasetCOCO
    from src.backbone_vitsplus import DinoBackboneV3
    if getattr(args, 'old_arch', False):
        # best_e199 (May-28 run) used the 192-ch / 3×3 head, NOT the current
        # 256-ch cosine-classifier rewrite. Use the reconstructed old module.
        from src.head_vitsplus_mix import DINODetectionHeadMix
        print('  [arch] using OLD head (model_head_mix_v199) for best_e199-style ckpt')
    else:
        from src.head_vitsplus_mix    import DINODetectionHeadMix
    from src.decode         import decode_outputs_OBB

    device = (torch.device(args.device) if args.device
              else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    ckpt_path    = args.checkpoint or cfg.MODEL_PATH_INFERENCE
    score_thresh = args.score_thresh if args.score_thresh > 0 \
                   else float(getattr(cfg, 'VAL_SCORE_THRESH', 0.2))
    nms_thresh   = args.nms_thresh if args.nms_thresh > 0 \
                   else float(getattr(cfg, 'VAL_NMS_THRESH', 0.6))
    rare_classes = {'traffic_light_red', 'traffic_light_green',
                    'traffic_light_other', 'warning_light', 'neubie', 'ev_open'}
    rare_thresh  = float(getattr(cfg, 'VAL_RARE_THRESH', 0.08))

    img_size = cfg.IMG_SIZE
    img_h, img_w = (int(img_size[0]), int(img_size[1])) \
                   if isinstance(img_size, (tuple, list)) else (int(img_size), int(img_size))
    img_size_hw = (img_h, img_w)

    layout = args.layout or getattr(cfg, 'DATASET_LAYOUT', 'custom')

    SEP = '═' * 68
    sep = '─' * 68

    print(SEP)
    print(f'  DINOv3 Mix Model Evaluation')
    print(SEP)
    print(f'  Config       : {args.config}')
    print(f'  Checkpoint   : {ckpt_path}')
    print(f'  Dataset      : {cfg.NEUBIE_ROOT}')
    print(f'  Split        : {args.split}')
    print(f'  Device       : {device}')
    print(f'  Img size     : {img_h}×{img_w}  layout={layout}')
    print(f'  Score thresh : {score_thresh}  (rare={rare_thresh})  [P/R/F1]')
    print(f'  AP thresh    : {args.ap_score_thresh}  [AP collection]')
    print(f'  NMS thresh   : {nms_thresh}')
    print(f'  IoU thresh   : {args.iou_thresh}')
    print(SEP)

    # ── Dataset (Neubie validation split) ─────────────────────────────────────
    dataset = DatasetCOCO(
        cfg.NEUBIE_ROOT, args.split, img_size_hw, cfg.PATCH_SIZE,
        augment_prob=0.0, mean=cfg.IMG_MEAN, std=cfg.IMG_STD,
        layout=layout,
    )
    num_classes = len(dataset.class_names)
    class_names = dataset.class_names
    print(f'  Classes : {num_classes}  |  Images : {len(dataset)}')
    print(SEP)

    if args.max_images > 0:
        indices = list(range(min(args.max_images, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f'  [INFO] Evaluating on first {len(dataset)} images (--max-images)')
        print(SEP)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=device.type == 'cuda')

    per_class_thresh = [rare_thresh if name in rare_classes else score_thresh
                        for name in class_names]

    # ── Model ─────────────────────────────────────────────────────────────────
    dino_model_name, dino_weights, embed_dim, n_layers = \
        infer_backbone_from_ckpt(ckpt_path, cfg)

    num_coco_classes, num_neubie_classes = infer_num_classes(ckpt_path)
    use_obb = infer_obb(ckpt_path)

    print(f'  Backbone     : {dino_model_name}  (embed={embed_dim}, layers={n_layers})')
    print(f'  Neubie cls   : {num_neubie_classes}  |  COCO cls (unused): {num_coco_classes}')
    print(f'  OBB branch   : {"yes" if use_obb else "no (axis-aligned)"}')

    if num_neubie_classes != num_classes:
        print(f'  [WARN] Checkpoint Neubie classes ({num_neubie_classes}) '
              f'!= dataset classes ({num_classes}). Check your checkpoint.')

    dino_model = torch.hub.load(
        repo_or_dir=cfg.DINOV3_DIR, model=dino_model_name,
        source='local', weights=dino_weights,
    )
    dino_backbone = DinoBackboneV3(dino_model, n_layers).to(device).eval()
    for param in dino_backbone.parameters():
        param.requires_grad = False

    model_head = DINODetectionHeadMix(
        backbone_out_channels=embed_dim,
        fpn_channels=cfg.FPN_CH,
        num_coco_classes=num_coco_classes,
        num_neubie_classes=num_neubie_classes,
        num_convs=cfg.N_CONVS,
        obb=use_obb,
    ).to(device).eval()

    if args.fuse:
        model_head.fuse()
        print('  RepDWSBlock branches fused.')

    load_checkpoint(ckpt_path, model_head)
    print(SEP)

    # ── Inference loop ────────────────────────────────────────────────────────
    image_results_ap  = []
    image_results_prf = []

    ap_thresh_list = [args.ap_score_thresh] * num_classes

    t0 = time.time()

    with torch.no_grad():
        for images, batch_boxes, batch_labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device, dtype=torch.float)
            B      = images.shape[0]

            # DinoBackboneV3 returns a list of 3 feature tensors
            feats   = dino_backbone(images)
            # Always use Neubie classifier at eval
            outputs = model_head(feats, dataset='neubie')

            feat_w  = outputs['cls'][0].shape[3]
            s0      = float(img_w / feat_w)
            strides = [s0 * (2 ** lvl) for lvl in range(len(outputs['cls']))]

            for bi in range(B):
                one_out = {k: [o[bi:bi+1] for o in v]
                           for k, v in outputs.items()}

                gt_boxes_abs = rel_xywh_to_abs_xyxy(
                    batch_boxes[bi].float(), img_h, img_w)
                gt_labels_t  = batch_labels[bi].long()

                # AP pass: low threshold to capture full PR curve
                pred_xywhr_ap, pred_scores_ap, pred_labels_ap = decode_outputs_OBB(
                    one_out, img_size_hw, strides=strides,
                    score_thresh=ap_thresh_list,
                    nms_thresh=nms_thresh,
                    max_detections=args.max_det,
                )
                image_results_ap.append({
                    'pred_boxes':  xywhr_to_xyxy_norot(pred_xywhr_ap.cpu()),
                    'pred_scores': pred_scores_ap.cpu(),
                    'pred_labels': pred_labels_ap.cpu().long(),
                    'gt_boxes':    gt_boxes_abs.cpu(),
                    'gt_labels':   gt_labels_t.cpu(),
                })

                # P/R/F1 pass: configured operating-point threshold
                pred_xywhr_prf, pred_scores_prf, pred_labels_prf = decode_outputs_OBB(
                    one_out, img_size_hw, strides=strides,
                    score_thresh=per_class_thresh,
                    nms_thresh=nms_thresh,
                    max_detections=args.max_det,
                )
                image_results_prf.append({
                    'pred_boxes':  xywhr_to_xyxy_norot(pred_xywhr_prf.cpu()),
                    'pred_scores': pred_scores_prf.cpu(),
                    'pred_labels': pred_labels_prf.cpu().long(),
                    'gt_boxes':    gt_boxes_abs.cpu(),
                    'gt_labels':   gt_labels_t.cpu(),
                })

    elapsed = time.time() - t0
    print(f'\n  Inference: {len(dataset)} images in {elapsed:.1f}s '
          f'({len(dataset)/elapsed:.1f} img/s)')

    # ── AP across IoU thresholds ──────────────────────────────────────────────
    iou_thresholds = np.arange(0.50, 1.00, 0.05).tolist()
    ap50_cls       = None
    ap5095_cls     = np.zeros(num_classes)
    n_gt_cls       = None
    iou_tp_all     = []

    for thr in iou_thresholds:
        ap_cls, n_gt, iou_tps = match_detections(
            image_results_ap, num_classes, iou_thresh=thr)
        if abs(thr - 0.50) < 1e-4:
            ap50_cls = ap_cls.copy()
            n_gt_cls = n_gt.copy()
            iou_tp_all = iou_tps
        ap5095_cls += np.nan_to_num(ap_cls)

    ap5095_cls /= len(iou_thresholds)

    prec, rec, f1, _ = precision_recall_f1_at_thresh(
        image_results_prf, num_classes, iou_thresh=args.iou_thresh)

    valid   = ~np.isnan(ap50_cls)
    mAP50   = float(np.mean(ap50_cls[valid]))   if valid.any() else 0.0
    mAP5095 = float(np.mean(ap5095_cls[valid])) if valid.any() else 0.0
    mIoU    = float(np.mean(iou_tp_all))        if iou_tp_all  else 0.0

    # ── Results table ─────────────────────────────────────────────────────────
    col_w = max(len(n) for n in class_names) + 2

    print(f'\n{SEP}')
    print(f'  {"Class":<{col_w}}  {"N_GT":>6}  '
          f'{"AP@50":>7}  {"AP@50-95":>9}  '
          f'{"Prec":>6}  {"Rec":>6}  {"F1":>6}')
    print(f'  {sep}')

    for c, name in enumerate(class_names):
        if n_gt_cls[c] == 0:
            ap50_s = '    n/a'; ap5095_s = '      n/a'
            p_s = '   n/a'; r_s = '   n/a'; f1_s = '   n/a'
        else:
            ap50_s   = f'{ap50_cls[c]:7.4f}'
            ap5095_s = f'{ap5095_cls[c]:9.4f}'
            p_s  = f'{prec[c]:6.3f}'
            r_s  = f'{rec[c]:6.3f}'
            f1_s = f'{f1[c]:6.3f}'

        tag = ' *' if name in rare_classes else '  '
        bar = '█' * int(float(ap50_s.strip() or 0) * 20) \
              if ap50_s.strip() != 'n/a' else ''
        print(f'  {name:<{col_w}}{tag} {n_gt_cls[c]:>6}  '
              f'{ap50_s}  {ap5095_s}  '
              f'{p_s}  {r_s}  {f1_s}  {bar}')

    print(f'\n{SEP}')
    print(f'  {"mAP @ IoU=0.50":<30}  {mAP50:>9.4f}')
    print(f'  {"mAP @ IoU=0.50:0.95":<30}  {mAP5095:>9.4f}')
    print(f'  {"mIoU (TP detections)":<30}  {mIoU:>9.4f}')
    print(f'  {"AP collection thresh":<30}  {args.ap_score_thresh:>9.4f}')
    print(f'  {"P/R/F1 thresh (default)":<30}  {score_thresh:>9.3f}')
    print(f'  {"P/R/F1 thresh (rare *)":<30}  {rare_thresh:>9.3f}')
    print(SEP)
    print(f'  (* = rare class, lower score threshold applied)')
    print(SEP)


if __name__ == '__main__':
    main()
