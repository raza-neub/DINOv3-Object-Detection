#!/usr/bin/env python3
"""Image-folder inference → time-synced videos.

Recursively finds image sequences under --input (each /images/ subfolder
becomes one output video), runs detection, and writes annotated .mp4 videos.

Usage:
    python inference/inference_images.py \
        --config config/config_convnext_small.py \
        --checkpoint results/mixed-dataset-training/cnx_small_phaseE/best.pth \
        --input visualization/neubie_dataset/ \
        --out-dir visualization/phaseE_small_issues \
        --device cuda:2 --fps 15
"""
from __future__ import annotations
import argparse, importlib.util, os, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cfg(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


def load_checkpoint(ckpt_path, model):
    sd = torch.load(ckpt_path, map_location='cpu')
    head_sd = sd.get('model_head', sd)
    ema_sd = sd.get('ema', None)
    if ema_sd:
        print(f'  Using EMA weights')
        head_sd = ema_sd
    missing, unexpected = model.load_state_dict(head_sd, strict=False)
    if missing:
        print(f'  [WARN] Missing keys ({len(missing)}): {missing[:3]} …')
    if unexpected:
        print(f'  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]} …')


COLORS = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (200, 200, 0), (200, 0, 200), (0, 200, 200), (128, 128, 255),
]


def draw_detections(img, boxes, scores, labels, class_names):
    for i in range(len(scores)):
        x1, y1, x2, y2 = boxes[i].int().tolist()
        cls_id = int(labels[i])
        score = float(scores[i])
        color = COLORS[cls_id % len(COLORS)]
        name = class_names[cls_id] if cls_id < len(class_names) else f'cls{cls_id}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f'{name} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, txt, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def find_sequences(input_dir: Path):
    """Find all /images/ subdirs, each becomes one video sequence."""
    img_dirs = sorted(input_dir.rglob('images'))
    sequences = []
    for d in img_dirs:
        if not d.is_dir():
            continue
        frames = sorted(d.glob('*.jpg')) + sorted(d.glob('*.png'))
        if frames:
            # Build a short name from the path for the output video
            rel = d.parent.relative_to(input_dir)
            name = str(rel).replace('/', '_').replace(' ', '_')
            sequences.append({'name': name, 'frames': sorted(frames)})
    return sequences


def parse_args():
    p = argparse.ArgumentParser('Image folder → video inference')
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--input', type=str, required=True,
                   help='Root directory (recursively finds /images/ subdirs)')
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--score-thresh', type=float, default=0.40)
    p.add_argument('--rare-thresh', type=float, default=0.30)
    p.add_argument('--nms-thresh', type=float, default=0.30)
    p.add_argument('--fps', type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device)

    input_dir = Path(args.input)
    sequences = find_sequences(input_dir)
    total_frames = sum(len(s['frames']) for s in sequences)
    print(f'Found {len(sequences)} sequences, {total_frames} total frames under {input_dir}')
    for s in sequences:
        print(f'  {s["name"]}: {len(s["frames"])} frames')

    if not sequences:
        print('No image sequences found.')
        return

    infer_h = int(cfg.IMG_SIZE[0]) if isinstance(cfg.IMG_SIZE, (tuple, list)) else int(cfg.IMG_SIZE)
    infer_w = int(cfg.IMG_SIZE[1]) if isinstance(cfg.IMG_SIZE, (tuple, list)) else int(cfg.IMG_SIZE)

    with open(cfg.CLASS_NAMES_PATH) as f:
        class_names = [ln.strip() for ln in f if ln.strip()]

    rare_classes = {'traffic_light_red', 'traffic_light_green',
                    'traffic_light_other', 'warning_light', 'neubie', 'ev_open'}
    per_class_thresh = [args.rare_thresh if n in rare_classes else args.score_thresh
                        for n in class_names]

    print(f'Thresholds: score={args.score_thresh} rare={args.rare_thresh} nms={args.nms_thresh}')
    print(f'Infer size: {infer_h}x{infer_w} | FPS: {args.fps}')

    # Load model
    from src.common import image_to_tensor
    from src.backbone_convnext import DinoBackboneConvNeXt
    from src.head_convnext import ConvNeXtDetectionHeadMixP2
    from src.decode import decode_outputs_aabb

    dino_raw = torch.hub.load(
        repo_or_dir=cfg.DINOV3_DIR, model=cfg.DINO_MODEL,
        source='local', weights=cfg.DINO_WEIGHTS)
    backbone = DinoBackboneConvNeXt(dino_raw).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    head = ConvNeXtDetectionHeadMixP2(
        in_channels_list=cfg.CONVNEXT_IN_CHANNELS,
        fpn_channels=cfg.FPN_CH,
        num_coco_classes=getattr(cfg, 'NUM_COCO_CLASSES', 80),
        num_neubie_classes=getattr(cfg, 'NUM_NEUBIE_CLASSES', len(class_names)),
        num_convs=cfg.N_CONVS,
        obb=getattr(cfg, 'USE_OBB', False),
        reg_max=getattr(cfg, 'DFL_REG_MAX', 16) if getattr(cfg, 'USE_DFL', False) else 0,
        use_centerness=getattr(cfg, 'USE_CENTERNESS', False),
        use_dcn=getattr(cfg, 'USE_DCN', False),
        cosine_cls=getattr(cfg, 'COSINE_CLS', True),
        use_cross_level_attn=getattr(cfg, 'USE_CROSS_LEVEL_ATTN', False),
        use_aux_decoder=False,
    ).to(device)
    load_checkpoint(args.checkpoint, head)

    class _Wrapper(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h
        def forward(self, feats):
            return self.h(feats, dataset='neubie')
    model_head = _Wrapper(head).to(device).eval()

    img_mean = np.array(getattr(cfg, 'IMG_MEAN', [0.485, 0.456, 0.406]), dtype=np.float32).reshape(3, 1, 1)
    img_std = np.array(getattr(cfg, 'IMG_STD', [0.229, 0.224, 0.225]), dtype=np.float32).reshape(3, 1, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0_all = time.time()
    total_processed = 0

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for si, seq in enumerate(sequences):
            frames = seq['frames']
            name = seq['name']

            # Read first frame to get resolution
            first = cv2.imread(str(frames[0]))
            orig_h, orig_w = first.shape[:2]
            sx = orig_w / infer_w
            sy = orig_h / infer_h

            out_path = out_dir / f'{name}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (orig_w, orig_h))

            t0_seq = time.time()
            for fi, fpath in enumerate(frames):
                img_bgr = cv2.imread(str(fpath))
                if img_bgr is None:
                    continue

                resized = cv2.resize(img_bgr, (infer_w, infer_h))
                img_t = image_to_tensor(resized, img_mean, img_std).unsqueeze(0).to(device)
                feats = backbone(img_t)
                outputs = model_head(feats)

                boxes, scores, labels = decode_outputs_aabb(
                    outputs, (infer_h, infer_w),
                    score_thresh=per_class_thresh,
                    nms_thresh=args.nms_thresh,
                    max_detections=300,
                )

                if len(boxes) > 0:
                    boxes[:, 0] *= sx; boxes[:, 2] *= sx
                    boxes[:, 1] *= sy; boxes[:, 3] *= sy

                vis = draw_detections(img_bgr.copy(), boxes, scores, labels, class_names)
                writer.write(vis)

            writer.release()
            elapsed = time.time() - t0_seq
            total_processed += len(frames)
            print(f'[{si+1}/{len(sequences)}] {name}: {len(frames)} frames → {out_path.name} '
                  f'({elapsed:.1f}s, {len(frames)/elapsed:.1f} fps)')

    total_elapsed = time.time() - t0_all
    print(f'\nDone: {total_processed} frames in {total_elapsed:.1f}s '
          f'({total_processed/total_elapsed:.1f} fps)')
    print(f'Output: {out_dir}')


if __name__ == '__main__':
    main()
