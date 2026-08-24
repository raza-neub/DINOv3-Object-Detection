#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch video inference for DINOv3 v2.

Phase 1 — Inference:
    Runs detection on every video in ALL_VIDEOS, writes individual output mp4s
    to <out_dir>/<stem>.mp4.  Already-processed videos are skipped (use --force
    to re-run).

Phase 2 — Tiling:
    Groups consecutive output videos N at a time (default N=3) and writes
    side-by-side tiled videos to <out_dir>/tile_group_<i>.mp4.
    Also writes a single grid video <out_dir>/grid_all.mp4 when all videos fit
    in a regular grid.

Run:
    python inference/inference_batch_v2.py --config config/config_v2.py \\
        --device cuda:0 --tile 3 --aabb --show-fps

    # Custom video list
    python inference/inference_batch_v2.py --config config/config_v2.py \\
        --videos /path/a.mp4 /path/b.mp4 /path/c.mp4

    # Skip inference, only re-tile already-processed videos
    python inference/inference_batch_v2.py --config config/config_v2.py \\
        --tile-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Default video list ─────────────────────────────────────────────────────────
_TRAFFIC = '/media/data/jb/data/jeongbae/repo/bag_to_mp4/traffic/video_1'
ALL_VIDEOS = [
    f'{_TRAFFIC}/sensor_data_111905_55.mp4',
    f'{_TRAFFIC}/sensor_data_112105_57.mp4',
    f'{_TRAFFIC}/sensor_data_112305_59.mp4',
    f'{_TRAFFIC}/sensor_data_113705_73.mp4',
    f'{_TRAFFIC}/sensor_data_113805_74.mp4',
    f'{_TRAFFIC}/sensor_data_113905_75.mp4',
    f'{_TRAFFIC}/sensor_data_114305_79.mp4',
    '/home/raza/Raza/neubi/DEIMv2/output.mp4',
    '/home/raza/Raza/neubi/DEIMv2/sample_video_original.mp4',
]


# ── Config / backbone helpers (same as inference_video_v2.py) ──────────────────

def load_cfg(path: str):
    spec = importlib.util.spec_from_file_location('cfg', str(Path(path).resolve()))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HASH = {
    'dinov3_vits16':     '08c60483',
    'dinov3_vits16plus': '4057cbaa',
    'dinov3_vitb16':     '73cec8be',
    'dinov3_vitl16':     '8aa4cbdd',
    'dinov3_vith16plus': '7c1da9a5',
    'dinov3_vit7b16':    'a955f4ea',
}


def infer_backbone(ckpt_path: str, cfg_mod) -> tuple:
    """Return (model_name, weights_path, embed_dim, n_layers)."""
    run_cfg_file = Path(ckpt_path).parent / 'run_config.json'
    if run_cfg_file.exists():
        try:
            rc = json.loads(run_cfg_file.read_text())
            m, w = rc.get('DINO_MODEL', ''), rc.get('DINO_WEIGHTS', '')
            if m and w:
                ed = cfg_mod.MODEL_TO_EMBED_DIM.get(m, 768)
                nl = cfg_mod.MODEL_TO_NUM_LAYERS.get(m, 12)
                return m, w, int(ed), int(nl)
        except Exception:
            pass
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd   = ckpt.get('model_head', ckpt.get('state_dict', ckpt)) \
               if isinstance(ckpt, dict) else ckpt
        # v3/mix checkpoint key
        wt = sd.get('fpn.proj_shallow.0.weight') or sd.get('fpn.proj.0.weight')
        if wt is not None:
            ed  = int(wt.shape[1])
            d2m = {v: k for k, v in cfg_mod.MODEL_TO_EMBED_DIM.items()}
            m   = d2m.get(ed, cfg_mod.DINO_MODEL)
            nl  = cfg_mod.MODEL_TO_NUM_LAYERS.get(m, 12)
            h   = _HASH.get(m, '')
            wdir = str(Path(cfg_mod.DINO_WEIGHTS).parent)
            w   = str(Path(wdir) / f'{m}_pretrain_lvd1689m-{h}.pth') if h \
                  else cfg_mod.DINO_WEIGHTS
            return m, w, ed, int(nl)
    except Exception:
        pass
    m = cfg_mod.DINO_MODEL
    return m, cfg_mod.DINO_WEIGHTS, cfg_mod.MODEL_TO_EMBED_DIM[m], cfg_mod.MODEL_TO_NUM_LAYERS[m]


def sniff_head_type(ckpt_path: str) -> str:
    """Return 'convnext', 'mix', 'v3', or 'v2' based on state-dict keys."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        sd   = ckpt.get('ema', ckpt.get('model_head', ckpt)) \
               if isinstance(ckpt, dict) else ckpt
        if isinstance(sd, dict):
            # ConvNeXt head has per-stage projections (proj_s8, proj_s16, proj_s32)
            if 'fpn.proj_s8.0.weight' in sd:
                return 'convnext'
            if 'head.cls_neubie.0.weight' in sd:
                return 'mix'
            if 'fpn.proj_shallow.0.weight' in sd:
                return 'v3'
    except Exception:
        pass
    return 'v2'


class _MixHeadWrapper(torch.nn.Module):
    """Wraps DINODetectionHeadMix to always forward with dataset='neubie'."""
    def __init__(self, head: torch.nn.Module):
        super().__init__()
        self.head = head

    def forward(self, feats):
        return self.head(feats, dataset='neubie')

    def eval(self):
        self.head.eval()
        return super().eval()

    def train(self, mode=True):
        self.head.train(mode)
        return super().train(mode)


def load_checkpoint(path: str, model_head: torch.nn.Module):
    ckpt = torch.load(path, map_location='cpu')
    sd   = ckpt.get('model_head', ckpt.get('state_dict', ckpt)) \
           if isinstance(ckpt, dict) else ckpt
    sd   = {k: v for k, v in sd.items()
            if not k.endswith('total_ops') and not k.endswith('total_params')}
    missing, unexpected = model_head.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [WARN] Missing keys ({len(missing)}): {missing[:3]} …')
    if unexpected:
        print(f'  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …')


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _color(cls_id: int) -> tuple:
    hue = int((cls_id * 137.508) % 180)
    bgr = cv2.cvtColor(np.array([[[hue, 220, 255]]], dtype=np.uint8),
                       cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_aabb(frame, boxes_xywhr, scores, labels, class_names):
    if boxes_xywhr is None or boxes_xywhr.numel() == 0:
        return frame
    b = boxes_xywhr.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    l = labels.detach().cpu().numpy()
    out = frame.copy()
    for i in range(len(l)):
        cid   = int(l[i])
        name  = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        text  = f'{name} {float(s[i]):.2f}'
        color = _color(cid)
        cx, cy, w, h = b[i, :4]
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw, ty), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        cv2.putText(out, text, (x1, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0) if lum > 128 else (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_obb(frame, boxes_xywhr, scores, labels, class_names):
    from src.decode import xywhr2xyxyxyxy
    if boxes_xywhr is None or boxes_xywhr.numel() == 0:
        return frame
    corners = xywhr2xyxyxyxy(boxes_xywhr.detach().cpu()).numpy()
    s = scores.detach().cpu().numpy()
    l = labels.detach().cpu().numpy()
    out = frame.copy()
    for i in range(len(l)):
        cid   = int(l[i])
        name  = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        text  = f'{name} {float(s[i]):.2f}'
        color = _color(cid)
        pts   = corners[i].astype(np.int32)
        cv2.drawContours(out, [pts], 0, color, 2)
        tx, ty = int(pts[:, 0].min()), int(pts[:, 1].min())
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty_l = max(ty - 4, th + 4)
        cv2.rectangle(out, (tx, ty_l - th - 4), (tx + tw, ty_l), color, -1)
        lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
        cv2.putText(out, text, (tx, ty_l - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0) if lum > 128 else (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── SAHI frame detect (upper-strip tiling for far traffic lights) ───────────────

def _sahi_detect_frame(model_head, dino_backbone, img_t, img_hw, *,
                       per_class_thresh, tl_ids, nms_thresh, topk,
                       upper_frac=0.55, cols=2, overlap=0.2):
    """Full-frame (all classes) + upper-strip tiled traffic-light dets, merged.

    img_t : (1,3,H,W) normalised input. Returns (boxes_xywhr, scores, labels) in
    (H,W) model space. A "tile" is a crop of the upper strip up-sampled back to
    (H,W), so far lights are magnified; only traffic_light_* dets are kept from
    tiles and merged into the full-frame result with per-class NMS.
    """
    from src.decode import detection_inference_OBB, rotated_nms_per_class
    H, W = img_hw
    fb, fs, fl = detection_inference_OBB(
        model_head, dino_backbone(img_t), img_hw,
        per_class_thresh=per_class_thresh, nms_thresh=nms_thresh, max_detections=topk)
    boxes_all, scores_all, labels_all = [fb], [fs], [fl]

    tl_set = set(tl_ids or [])
    y1 = int(round(H * upper_frac))
    tw = W / (cols - (cols - 1) * overlap)
    step = tw * (1.0 - overlap)
    for i in range(cols):
        x0 = int(round(i * step)); x1 = int(round(min(x0 + tw, W)))
        if x1 - x0 < 8:
            continue
        crop    = img_t[:, :, 0:y1, x0:x1]
        crop_up = F.interpolate(crop, size=(H, W), mode='bilinear', align_corners=False)
        tb, ts, tlbl = detection_inference_OBB(
            model_head, dino_backbone(crop_up), img_hw,
            per_class_thresh=per_class_thresh, nms_thresh=nms_thresh, max_detections=topk)
        if tb.shape[0] == 0:
            continue
        keep = torch.tensor([int(c) in tl_set for c in tlbl.tolist()],
                            dtype=torch.bool, device=tb.device)
        if keep.sum() == 0:
            continue
        tb, ts, tlbl = tb[keep], ts[keep], tlbl[keep]
        sx = (x1 - x0) / float(W); sy = y1 / float(H)   # crop→model scale
        tb = tb.clone()
        tb[:, 0] = tb[:, 0] * sx + x0     # cx
        tb[:, 1] = tb[:, 1] * sy          # cy (crop starts at y=0)
        tb[:, 2] = tb[:, 2] * sx          # w
        tb[:, 3] = tb[:, 3] * sy          # h
        boxes_all.append(tb); scores_all.append(ts); labels_all.append(tlbl)

    boxes  = torch.cat(boxes_all, 0); scores = torch.cat(scores_all, 0)
    labels = torch.cat(labels_all, 0)
    if boxes.shape[0] == 0:
        return boxes, scores, labels
    keep = rotated_nms_per_class(boxes, scores, labels, nms_thresh, max_detections=topk)
    return boxes[keep], scores[keep], labels[keep]


# ── Phase 1: Inference ─────────────────────────────────────────────────────────

def run_inference_one(video_in: str, video_out: str, *,
                      dino_backbone, model_head, image_to_tensor_fn,
                      img_mean, img_std, infer_h, infer_w,
                      score_thresh, nms_thresh, per_class_thresh,
                      class_names, topk, device, aabb: bool, show_fps: bool,
                      sahi: bool = False, tl_ids=None,
                      sahi_upper_frac: float = 0.55, sahi_cols: int = 2,
                      sahi_overlap: float = 0.2):
    from src.decode import detection_inference_OBB

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f'  [SKIP] Cannot open: {video_in}')
        return False

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  {Path(video_in).name}: {orig_w}×{orig_h} @ {fps:.1f} FPS  ({n_total} frames)')

    Path(video_out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (orig_w, orig_h))

    sx = orig_w / infer_w
    sy = orig_h / infer_h
    frame_idx = 0
    t0 = time.time()

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resized   = cv2.resize(frame_rgb, (infer_w, infer_h),
                                   interpolation=cv2.INTER_LINEAR)
            img_t = image_to_tensor_fn(resized, img_mean, img_std) \
                    .unsqueeze(0).to(device)

            if sahi:
                boxes, scores, labs = _sahi_detect_frame(
                    model_head, dino_backbone, img_t, (infer_h, infer_w),
                    per_class_thresh=per_class_thresh, tl_ids=tl_ids,
                    nms_thresh=nms_thresh, topk=topk,
                    upper_frac=sahi_upper_frac, cols=sahi_cols, overlap=sahi_overlap,
                )
            else:
                feat = dino_backbone(img_t)
                boxes, scores, labs = detection_inference_OBB(
                    model_head, feat, (infer_h, infer_w),
                    score_thresh=score_thresh, nms_thresh=nms_thresh,
                    per_class_thresh=per_class_thresh, max_detections=topk,
                )

            # Scale boxes back to original resolution
            if boxes is not None and boxes.numel() > 0:
                scaled = boxes.clone()
                scaled[:, 0] *= sx; scaled[:, 1] *= sy
                scaled[:, 2] *= sx; scaled[:, 3] *= sy
            else:
                scaled = boxes

            draw_fn = draw_aabb if aabb else draw_obb
            vis = draw_fn(frame_bgr, scaled, scores, labs, class_names)
            writer.write(vis)
            frame_idx += 1

            if show_fps and frame_idx % 50 == 0:
                elapsed = time.time() - t0
                print(f'    frame {frame_idx}/{n_total}  '
                      f'{frame_idx/elapsed:.1f} FPS avg', flush=True)

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f'  → saved {video_out}  ({frame_idx} frames, {elapsed:.1f}s, '
          f'{frame_idx/elapsed:.1f} FPS)', flush=True)
    return True


# ── Phase 2: Tiling ────────────────────────────────────────────────────────────

def tile_videos(video_paths: list[str], out_path: str,
                tile_w: int = 640, tile_h: int = 360,
                label: bool = True) -> bool:
    """Tile N videos side by side into one output video.

    All videos are resized to tile_w×tile_h.  The shorter video freezes on
    its last frame.  Output FPS = FPS of the first video.
    """
    caps = [cv2.VideoCapture(p) for p in video_paths]
    valid = [(i, c) for i, c in enumerate(caps) if c.isOpened()]
    if not valid:
        print(f'  [SKIP] No valid videos for tile: {[Path(p).name for p in video_paths]}')
        for c in caps: c.release()
        return False

    fps    = caps[valid[0][0]].get(cv2.CAP_PROP_FPS) or 30.0
    n_cols = len(valid)
    out_w  = tile_w * n_cols
    out_h  = tile_h

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (out_w, out_h))

    last_frames = [None] * len(caps)
    frame_idx   = 0
    names       = [Path(p).stem[:24] for p in video_paths]

    print(f'  Tiling {n_cols} videos → {Path(out_path).name}  '
          f'({out_w}×{out_h} @ {fps:.1f} FPS)', flush=True)

    while True:
        tiles = []
        any_new = False
        for i, cap in enumerate(caps):
            if not cap.isOpened():
                frame = last_frames[i]
            else:
                ret, frame = cap.read()
                if ret:
                    any_new = True
                    last_frames[i] = frame
                else:
                    cap.release()
                    frame = last_frames[i]

            if frame is None:
                frame = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (tile_w, tile_h))

            # Label each tile with the video name
            if label:
                cv2.rectangle(frame, (0, 0), (len(names[i]) * 9 + 6, 22),
                              (0, 0, 0), -1)
                cv2.putText(frame, names[i], (3, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(frame)

        if not any_new:
            break

        row = np.concatenate(tiles, axis=1)
        writer.write(row)
        frame_idx += 1

    for cap in caps:
        if cap.isOpened():
            cap.release()
    writer.release()
    print(f'  → {out_path}  ({frame_idx} frames)', flush=True)
    return True


def make_grid(video_paths: list[str], out_path: str,
              n_cols: int = 3, tile_w: int = 640, tile_h: int = 360) -> bool:
    """Arrange N videos in a grid of n_cols columns."""
    n_rows = math.ceil(len(video_paths) / n_cols)
    padded = list(video_paths) + [''] * (n_rows * n_cols - len(video_paths))
    groups = [padded[i * n_cols: (i + 1) * n_cols] for i in range(n_rows)]

    # Write temporary row videos, then stack them vertically
    tmp_rows = []
    for ri, grp in enumerate(groups):
        tmp = str(Path(out_path).parent / f'_tmp_row{ri}.mp4')
        tile_videos([p for p in grp if p], tmp, tile_w=tile_w, tile_h=tile_h)
        tmp_rows.append(tmp)

    # Stack rows vertically (all have same width = n_cols * tile_w)
    caps   = [cv2.VideoCapture(r) for r in tmp_rows]
    fps    = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    out_w  = n_cols * tile_w
    out_h  = n_rows * tile_h
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (out_w, out_h))

    last_frames = [None] * len(caps)
    print(f'  Building grid {n_cols}×{n_rows} → {Path(out_path).name}  '
          f'({out_w}×{out_h})', flush=True)

    while True:
        rows = []
        any_new = False
        for i, cap in enumerate(caps):
            ret, frame = (cap.read() if cap.isOpened() else (False, None))
            if ret:
                any_new = True
                last_frames[i] = frame
            else:
                cap.release()
                frame = last_frames[i]
            if frame is None:
                frame = np.zeros((tile_h, out_w, 3), dtype=np.uint8)
            rows.append(frame)
        if not any_new:
            break
        writer.write(np.concatenate(rows, axis=0))

    for cap in caps:
        if cap.isOpened(): cap.release()
    writer.release()

    # Clean up temp files
    for tmp in tmp_rows:
        try: Path(tmp).unlink()
        except Exception: pass

    print(f'  → {out_path}', flush=True)
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('DINOv3 v2 batch video inference')
    p.add_argument('--config',      type=str, required=True)
    p.add_argument('--videos',      nargs='+', default=[],
                   help='Override default video list')
    p.add_argument('--device',      type=str, default='')
    p.add_argument('--checkpoint',  type=str, default='',
                   help='Override config MODEL_PATH_INFERENCE')
    p.add_argument('--out-dir',     type=str, default='',
                   help='Output directory (default: visualization/v2/batch)')
    p.add_argument('--tile',        type=int, default=3,
                   help='Videos per tile group (default 3; use 2 for pairs)')
    p.add_argument('--tile-w',      type=int, default=640,
                   help='Width of each tile panel (default 640)')
    p.add_argument('--tile-h',      type=int, default=360,
                   help='Height of each tile panel (default 360)')
    p.add_argument('--aabb',        action='store_true',
                   help='Draw axis-aligned boxes (no rotation)')
    p.add_argument('--fuse',        action='store_true',
                   help='Fuse RepDWSBlock branches')
    p.add_argument('--show-fps',    action='store_true')
    p.add_argument('--force',       action='store_true',
                   help='Re-run inference even if output already exists')
    p.add_argument('--tile-only',   action='store_true',
                   help='Skip inference, only (re-)create tiled videos')
    p.add_argument('--no-grid',     action='store_true',
                   help='Skip the combined grid_all.mp4')
    p.add_argument('--infer-h',     type=int, default=-1)
    p.add_argument('--infer-w',     type=int, default=-1)
    # SAHI (upper-strip tiling) — boosts far traffic-light recall at ~(1+cols)× cost
    p.add_argument('--sahi',        action='store_true',
                   help='Enable SAHI upper-strip tiling for traffic_light_* classes')
    p.add_argument('--sahi-upper-frac', type=float, default=0.55,
                   help='Fraction of frame height (from top) to tile (default 0.55)')
    p.add_argument('--sahi-cols',   type=int, default=2,
                   help='Number of horizontal tiles over the upper strip (default 2)')
    p.add_argument('--sahi-overlap', type=float, default=0.2,
                   help='Tile overlap fraction (default 0.2)')
    # Score threshold overrides (override config values)
    p.add_argument('--score-thresh', type=float, default=-1,
                   help='Override SCORE_THRESH from config (default classes)')
    p.add_argument('--rare-thresh',  type=float, default=-1,
                   help='Override VAL_RARE_THRESH from config (rare classes)')
    p.add_argument('--nms-thresh',   type=float, default=-1,
                   help='Override NMS_THRESH from config')
    p.add_argument('--thresholds-json', type=str, default='',
                   help='Per-class thresholds JSON from calibrate_thresholds.py')
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    videos = args.videos if args.videos else ALL_VIDEOS
    videos = [v for v in videos if v]  # drop empty strings

    out_dir = Path(args.out_dir) if args.out_dir \
              else ROOT / 'visualization' / 'v2' / 'batch'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Inference ─────────────────────────────────────────────────────
    if not args.tile_only:
        from src.common import image_to_tensor

        ckpt_path = args.checkpoint or cfg.MODEL_PATH_INFERENCE
        device    = (torch.device(args.device) if args.device
                     else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        infer_h = args.infer_h if args.infer_h > 0 else \
                  (int(cfg.IMG_SIZE[0]) if isinstance(cfg.IMG_SIZE, (tuple, list))
                   else int(cfg.IMG_SIZE))
        infer_w = args.infer_w if args.infer_w > 0 else \
                  (int(cfg.IMG_SIZE[1]) if isinstance(cfg.IMG_SIZE, (tuple, list))
                   else int(cfg.IMG_SIZE))

        score_thresh = args.score_thresh if args.score_thresh > 0 \
                       else float(getattr(cfg, 'SCORE_THRESH', 0.2))
        nms_thresh   = args.nms_thresh if args.nms_thresh > 0 \
                       else float(getattr(cfg, 'NMS_THRESH', 0.6))
        rare_thresh  = args.rare_thresh if args.rare_thresh > 0 \
                       else float(getattr(cfg, 'VAL_RARE_THRESH', 0.08))
        topk         = int(getattr(cfg, 'TOPK_INFER', 300))
        print(f'Thresholds : score={score_thresh}  rare={rare_thresh}  nms={nms_thresh}')

        with open(cfg.CLASS_NAMES_PATH) as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

        # Per-class thresholds: from JSON (calibrated) or flat rare/default split
        if args.thresholds_json:
            import json as _json
            with open(args.thresholds_json) as _f:
                _thr = _json.load(_f)
            per_class_thresh = _thr['thresholds_list']
            assert len(per_class_thresh) == len(class_names), (
                f'thresholds_list has {len(per_class_thresh)} entries, '
                f'expected {len(class_names)}')
            print(f'Loaded per-class thresholds from {args.thresholds_json}')
            for _i, _n in enumerate(class_names):
                print(f'  {_n:<30} τ={per_class_thresh[_i]:.4f}')
        else:
            rare_classes = {'traffic_light_red', 'traffic_light_green',
                            'traffic_light_other', 'warning_light', 'neubie', 'ev_open'}
            per_class_thresh = [rare_thresh if n in rare_classes else score_thresh
                                for n in class_names]
        # Traffic-light class indices for SAHI (kept-from-tiles classes)
        tl_ids = [class_names.index(n) for n in
                  ('traffic_light_red', 'traffic_light_green', 'traffic_light_other')
                  if n in class_names]
        if args.sahi:
            print(f'SAHI enabled: upper_frac={args.sahi_upper_frac} cols={args.sahi_cols} '
                  f'overlap={args.sahi_overlap} → TL classes {tl_ids} '
                  f'(~{1+args.sahi_cols}× backbone/frame)')

        head_type = sniff_head_type(ckpt_path)

        # ConvNeXt backbone uses config directly (no infer_backbone heuristic)
        if head_type == 'convnext':
            dino_model_name = cfg.DINO_MODEL
            dino_weights    = cfg.DINO_WEIGHTS
            in_channels_list = cfg.CONVNEXT_IN_CHANNELS
            embed_dim = cfg.MODEL_TO_EMBED_DIM.get(dino_model_name, 1024)
            n_layers  = cfg.MODEL_TO_NUM_LAYERS.get(dino_model_name, 4)
        else:
            dino_model_name, dino_weights, embed_dim, n_layers = \
                infer_backbone(ckpt_path, cfg)

        print(f'Backbone : {dino_model_name}  (embed={embed_dim}, layers={n_layers})')
        print(f'Head type: {head_type}')
        print(f'Checkpoint: {ckpt_path}')
        print(f'Device   : {device}  |  Infer size: {infer_h}×{infer_w}')
        print(f'Classes  : {len(class_names)}  |  Videos: {len(videos)}\n')

        dino_raw = torch.hub.load(
            repo_or_dir=cfg.DINOV3_DIR, model=dino_model_name,
            source='local', weights=dino_weights,
        )

        if head_type == 'convnext':
            from src.backbone_convnext import DinoBackboneConvNeXt
            dino_backbone = DinoBackboneConvNeXt(dino_raw).to(device).eval()
        elif head_type in ('mix', 'v3'):
            from src.backbone_vitsplus import DinoBackboneV3
            dino_backbone = DinoBackboneV3(dino_raw, n_layers).to(device).eval()
        else:
            from src.backbone_vitsplus import DinoBackbone
            dino_backbone = DinoBackbone(dino_raw, n_layers).to(device).eval()
        for p in dino_backbone.parameters():
            p.requires_grad = False

        if head_type == 'convnext':
            from src.head_convnext import ConvNeXtDetectionHeadMixP2
            num_neubie = getattr(cfg, 'NUM_NEUBIE_CLASSES', len(class_names))
            num_coco   = getattr(cfg, 'NUM_COCO_CLASSES', 80)
            use_dfl    = getattr(cfg, 'USE_DFL', False)
            reg_max    = getattr(cfg, 'DFL_REG_MAX', 16) if use_dfl else 0
            use_dcn    = getattr(cfg, 'USE_DCN', False)
            use_obb    = getattr(cfg, 'USE_OBB', False)
            _head = ConvNeXtDetectionHeadMixP2(
                in_channels_list=in_channels_list,
                fpn_channels=cfg.FPN_CH,
                num_coco_classes=num_coco,
                num_neubie_classes=num_neubie,
                num_convs=cfg.N_CONVS,
                obb=use_obb,
                reg_max=reg_max,
                use_centerness=getattr(cfg, 'USE_CENTERNESS', False),
                use_dcn=use_dcn,
                use_aux_decoder=False,  # not needed at inference
            ).to(device)
            load_checkpoint(ckpt_path, _head)
            model_head = _MixHeadWrapper(_head).to(device).eval()
            print(f'  ConvNeXt head: in_channels={in_channels_list}  '
                  f'neubie_cls={num_neubie}, coco_cls={num_coco}, obb={use_obb}  '
                  f'dfl={use_dfl} dcn={use_dcn}')
        elif head_type == 'mix':
            # Infer class counts + architecture (3-level mix vs 4-level P2) from ckpt
            is_p2 = False
            has_obb = True
            try:
                _ckpt = torch.load(ckpt_path, map_location='cpu')
                _sd   = _ckpt.get('ema', _ckpt.get('model_head', _ckpt)) \
                        if isinstance(_ckpt, dict) else _ckpt
                num_neubie = int(_sd['head.cls_neubie.0.weight'].shape[0])
                num_coco   = int(_sd['head.cls_coco.0.weight'].shape[0])
                is_p2   = any(k.startswith('fpn.p2_refine') for k in _sd) \
                          or 'head.cls_neubie.3.weight' in _sd
                has_obb = 'head.angle_reg.0.weight' in _sd
            except Exception:
                num_neubie = getattr(cfg, 'NUM_NEUBIE_CLASSES', len(class_names))
                num_coco   = getattr(cfg, 'NUM_COCO_CLASSES', 80)
            if is_p2:
                from src.head_vitsplus_mix_p2 import DINODetectionHeadMixP2 as _MixHead
            else:
                from src.head_vitsplus_mix import DINODetectionHeadMix as _MixHead
            _head = _MixHead(
                backbone_out_channels=embed_dim,
                fpn_channels=cfg.FPN_CH,
                num_coco_classes=num_coco,
                num_neubie_classes=num_neubie,
                num_convs=cfg.N_CONVS,
                obb=has_obb,
            ).to(device)
            load_checkpoint(ckpt_path, _head)
            model_head = _MixHeadWrapper(_head).to(device).eval()
            print(f'  Mix head: {"P2 4-level" if is_p2 else "3-level"}  '
                  f'neubie_cls={num_neubie}, coco_cls={num_coco}, obb={has_obb}  '
                  f'(inference uses neubie classifier)')
        elif head_type == 'v3':
            from src.neck import DINODetectionHeadV3
            model_head = DINODetectionHeadV3(
                backbone_out_channels=embed_dim,
                fpn_channels=cfg.FPN_CH,
                num_classes=len(class_names),
                num_convs=cfg.N_CONVS,
                obb=True,
            ).to(device).eval()
            load_checkpoint(ckpt_path, model_head)
        else:
            from src.blocks import DINODetectionHead
            model_head = DINODetectionHead(
                backbone_out_channels=embed_dim,
                fpn_channels=cfg.FPN_CH,
                num_classes=len(class_names),
                num_convs=cfg.N_CONVS,
                obb=True,
            ).to(device).eval()
            if args.fuse:
                model_head.fuse()
            load_checkpoint(ckpt_path, model_head)

        img_mean = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
        img_std  = np.array(cfg.IMG_STD,  dtype=np.float32)[:, None, None]

        t_total = time.time()
        for vi, video_in in enumerate(videos):
            stem    = Path(video_in).stem
            out_mp4 = str(out_dir / f'{stem}.mp4')
            print(f'\n[{vi+1}/{len(videos)}] {stem}')
            if Path(out_mp4).exists() and not args.force:
                print(f'  [SKIP] Output already exists: {out_mp4}')
                continue
            run_inference_one(
                video_in, out_mp4,
                dino_backbone=dino_backbone, model_head=model_head,
                image_to_tensor_fn=image_to_tensor,
                img_mean=img_mean, img_std=img_std,
                infer_h=infer_h, infer_w=infer_w,
                score_thresh=score_thresh, nms_thresh=nms_thresh,
                per_class_thresh=per_class_thresh,
                class_names=class_names, topk=topk, device=device,
                aabb=args.aabb, show_fps=args.show_fps,
                sahi=args.sahi, tl_ids=tl_ids,
                sahi_upper_frac=args.sahi_upper_frac, sahi_cols=args.sahi_cols,
                sahi_overlap=args.sahi_overlap,
            )

        print(f'\nAll inference done in {time.time()-t_total:.1f}s\n')

    # ── Phase 2: Tiling ────────────────────────────────────────────────────────
    print('=' * 60)
    print('Phase 2: Tiling output videos')
    print('=' * 60)

    # Collect individual output mp4s (only those that exist, preserve order)
    output_videos = []
    for v in videos:
        mp4 = str(out_dir / f'{Path(v).stem}.mp4')
        if Path(mp4).exists():
            output_videos.append(mp4)
        else:
            print(f'  [WARN] Missing: {mp4}')

    if not output_videos:
        print('No output videos found — run without --tile-only first.')
        return

    n = args.tile
    groups = [output_videos[i:i+n] for i in range(0, len(output_videos), n)]
    tile_paths = []
    for gi, grp in enumerate(groups):
        names = '+'.join(Path(p).stem[-8:] for p in grp)
        out_tile = str(out_dir / f'tile_group_{gi:02d}_{names}.mp4')
        tile_videos(grp, out_tile, tile_w=args.tile_w, tile_h=args.tile_h)
        tile_paths.append(out_tile)

    if not args.no_grid and len(output_videos) > n:
        n_cols   = n
        grid_out = str(out_dir / f'grid_all_{len(output_videos)}x.mp4')
        make_grid(output_videos, grid_out, n_cols=n_cols,
                  tile_w=args.tile_w, tile_h=args.tile_h)

    print('\nDone.')
    print(f'  Individual : {out_dir}/<stem>.mp4')
    for tp in tile_paths:
        print(f'  Tile group : {tp}')
    if not args.no_grid and len(output_videos) > n:
        print(f'  Grid       : {out_dir}/grid_all_{len(output_videos)}x.mp4')


if __name__ == '__main__':
    main()
