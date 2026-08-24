# dataset_coco_v3.py — v3 augmentations + small-object augmentations
#   1. BGR/RGB bug fix: photometric_augment uses COLOR_RGB2HSV / COLOR_HSV2RGB
#   2. Mosaic augmentation: 4-image tiling (primary YOLO-style augmentation)
#   3. Random horizontal flip
#   4. Random translate
#   5. use_mosaic attribute: toggleable per-epoch (set False for last N epochs)
#
#   NEW (small-object levers, all config-gated, DEFAULT OFF → existing runs unchanged):
#   6. RandomZoomOut   — shrink image into a larger mean-padded canvas → more
#                        small-scale training instances (SSD "expand").
#   7. Copy-paste      — paste extra instances of target rare classes (bollard,
#                        traffic_light_*, scooter, warning_light) into the image.
#   8. SAHI-crop       — crop an upper-strip tile and up-sample it back to IMG_SIZE,
#                        mirroring inference-time SAHI geometry (upper_frac/cols/overlap)
#                        so train and SAHI-inference see the same magnified distribution.
#
#   BUG FIX: box clipping now computes x2=x1+w BEFORE clamping (the old order clamped
#   x1 first, then x2=clamp(x1_clamped+w), which widened boxes that hung off an edge),
#   and applies a visibility filter (drop boxes that lose too much area to a crop).

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


def _clip_boxes_xywh(boxes: torch.Tensor, labels: torch.Tensor,
                     min_vis: float = 0.0, min_wh: float = 1e-3):
    """Clip normalized-xywh boxes to [0,1] correctly and filter by visibility.

    boxes  : (N,4) normalized [x, y, w, h]  (x,y = top-left)
    min_vis: keep a box only if (clipped area / original area) >= min_vis
    min_wh : also drop boxes whose clipped w or h falls below this

    Correct order: x2 = x1 + w (from ORIGINAL coords) → clip both x1,x2 → w = x2-x1.
    """
    if boxes.numel() == 0:
        return boxes, labels
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    orig_area = (boxes[:, 2] * boxes[:, 3]).clamp(min=1e-9)

    cx1 = x1.clamp(0.0, 1.0)
    cy1 = y1.clamp(0.0, 1.0)
    cx2 = x2.clamp(0.0, 1.0)
    cy2 = y2.clamp(0.0, 1.0)
    nw = (cx2 - cx1).clamp(min=0.0)
    nh = (cy2 - cy1).clamp(min=0.0)

    out = torch.stack([cx1, cy1, nw, nh], dim=1)
    vis = (nw * nh) / orig_area
    keep = (nw > min_wh) & (nh > min_wh) & (vis >= min_vis)
    return out[keep], labels[keep]


class DatasetCOCOv3(Dataset):
    """
    COCO-format dataset loader with v3 + small-object augmentations.

    Layout (custom):
      <root>/train/images/ + <root>/train/annotations.json
      <root>/val/images/   + <root>/val/annotations.json
    """

    def __init__(self, root_dir, mode, img_size, patch_size,
                 augment_prob=0.0,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 layout='custom',
                 use_mosaic=True,
                 mosaic_prob=0.5,
                 hflip_prob=0.5,
                 translate_prob=0.5,
                 translate_max=0.1,
                 hue_delta=10,
                 saturation_range=(0.8, 1.2),
                 # ── NEW small-object augs (default OFF) ──
                 zoomout_prob=0.0,
                 zoomout_max_scale=2.0,
                 copypaste_prob=0.0,
                 copypaste_max_per_img=3,
                 copypaste_classes=(),
                 copypaste_scale_range=(0.6, 1.4),
                 sahicrop_prob=0.0,
                 sahicrop_upper_frac=0.55,
                 sahicrop_cols=2,
                 sahicrop_overlap=0.2,
                 sahicrop_min_vis=0.3):
        super().__init__()
        self.root_dir     = root_dir
        self.mode         = mode
        self.img_size     = img_size
        self.patch_size   = patch_size
        self.augment_prob = float(augment_prob)

        self.mean = np.array(mean, dtype=np.float32)[:, None, None]
        self.std  = np.array(std,  dtype=np.float32)[:, None, None]
        # mean fill colour for zoom-out canvas (0-255 RGB)
        self._fill_rgb = tuple(int(round(c * 255)) for c in mean)

        # Geometric augment params
        self.use_mosaic    = use_mosaic and (mode == 'train')
        self.mosaic_prob   = float(mosaic_prob)
        self.hflip_prob    = float(hflip_prob)
        self.translate_prob  = float(translate_prob)
        self.translate_max   = float(translate_max)
        # Color jitter — hue/saturation. For traffic-light colour classes,
        # hue jitter trains hue-invariance (counterproductive, since the label
        # IS the hue), so config sets hue_delta low (0–3). OpenCV H is 0–179.
        self.hue_delta       = float(hue_delta)
        self.saturation_range = tuple(saturation_range)

        # ── NEW small-object aug params ──
        self.zoomout_prob          = float(zoomout_prob)
        self.zoomout_max_scale     = float(zoomout_max_scale)
        self.copypaste_prob        = float(copypaste_prob)
        self.copypaste_max_per_img = int(copypaste_max_per_img)
        self.copypaste_classes     = tuple(copypaste_classes)
        self.copypaste_scale_range = tuple(copypaste_scale_range)
        self.sahicrop_prob         = float(sahicrop_prob)
        self.sahicrop_upper_frac   = float(sahicrop_upper_frac)
        self.sahicrop_cols         = int(sahicrop_cols)
        self.sahicrop_overlap      = float(sahicrop_overlap)
        self.sahicrop_min_vis      = float(sahicrop_min_vis)
        # classes whose copy-pasted instances should be biased to the upper image
        # region (lights mounted high); everything else biased to the lower region.
        self._cp_upper = {'traffic_light_red', 'traffic_light_green',
                          'traffic_light_other', 'warning_light'}

        if mode not in ('train', 'val'):
            raise ValueError("mode must be 'train' or 'val'")

        if layout == 'coco':
            split = 'train2017' if mode == 'train' else 'val2017'
            self.data_dir         = os.path.join(root_dir, 'images', split)
            self.path_annotations = os.path.join(
                root_dir, 'annotations', f'instances_{split}.json')
        else:  # 'custom'
            self.data_dir         = root_dir   # file_name already encodes split/images/...
            self.path_annotations = os.path.join(root_dir, mode, 'annotations.json')

        if not os.path.exists(self.path_annotations):
            raise FileNotFoundError(f'Annotation file not found: {self.path_annotations}')

        self.coco = COCO(self.path_annotations)
        self.ids  = list(sorted(self.coco.imgs.keys()))

        cats = sorted(self.coco.loadCats(self.coco.getCatIds()), key=lambda c: c['id'])
        self.class_names    = [cat['name'] for cat in cats]
        self.catid_to_label = {cat['id']: i for i, cat in enumerate(cats)}
        self.label_to_catid = {i: cat['id'] for i, cat in enumerate(cats)}

        # Build copy-paste source pool (train only, and only for target classes
        # that actually exist in THIS dataset — COCO won't have bollard/scooter,
        # so its pool stays empty and copy-paste is a silent no-op there).
        self._cp_pool = []
        if mode == 'train' and self.copypaste_prob > 0 and self.copypaste_classes:
            self._build_copypaste_pool()

        print(f'DatasetCOCOv3 [{mode}]: {len(self.ids)} images, '
              f'{len(self.class_names)} classes, mosaic={self.use_mosaic}'
              + (f', copy-paste pool={len(self._cp_pool)} instances'
                 if self._cp_pool else ''))

    # ── helpers ───────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.ids)

    def _get_hw(self):
        if isinstance(self.img_size, (tuple, list)):
            return int(self.img_size[0]), int(self.img_size[1])
        S = int(self.img_size)
        return S, S

    def _resize_rgb(self, image_rgb: np.ndarray) -> np.ndarray:
        H, W = self._get_hw()
        return cv2.resize(image_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    def image_to_tensor(self, image_rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 → normalized float CHW tensor."""
        img = image_rgb.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = (img - self.mean) / self.std
        return torch.from_numpy(img).float()

    def _img_path(self, img_info):
        return os.path.join(self.data_dir, img_info['file_name'])

    # ── raw load (no resize, no augment) ─────────────────────────────────────

    def _load_raw(self, idx):
        """Load RGB image + normalized-xywh boxes + labels.  No resize, no augment."""
        img_id   = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self._img_path(img_info)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to read image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]

        ann_ids     = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in annotations:
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x / w0, y / h0, w / w0, h / h0])
            labels.append(self.catid_to_label[ann['category_id']])

        if boxes:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)

        return image, boxes_t, labels_t

    # ── copy-paste source pool ──────────────────────────────────────────────

    def _build_copypaste_pool(self):
        """Index every target-class instance: (img_path, abs-bbox, label, src_w, src_h)."""
        target_labels = {self.catid_to_label[c['id']]: c['name']
                         for c in self.coco.loadCats(self.coco.getCatIds())
                         if c['name'] in self.copypaste_classes}
        if not target_labels:
            return
        for img_id in self.ids:
            info = self.coco.loadImgs(img_id)[0]
            w0, h0 = info.get('width'), info.get('height')
            if not w0 or not h0:
                continue
            path = self._img_path(info)
            for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
                if ann.get('iscrowd', 0):
                    continue
                lab = self.catid_to_label.get(ann['category_id'])
                if lab not in target_labels:
                    continue
                x, y, w, h = ann['bbox']
                if w < 4 or h < 4:          # skip degenerate / sub-4px source crops
                    continue
                self._cp_pool.append((path, (float(x), float(y), float(w), float(h)),
                                      int(lab), int(w0), int(h0)))

    # ── mosaic ────────────────────────────────────────────────────────────────

    def _mosaic(self, idx):
        """4-image 2×2 mosaic at the target resolution."""
        H, W  = self._get_hw()
        hH, hW = H // 2, W // 2

        indices = [idx] + random.choices(range(len(self)), k=3)

        mosaic_img  = np.zeros((H, W, 3), dtype=np.uint8)
        all_boxes   = []
        all_labels  = []

        for i, ind in enumerate(indices):
            img, boxes, labels = self._load_raw(ind)
            cell = cv2.resize(img, (hW, hH), interpolation=cv2.INTER_LINEAR)

            row, col = divmod(i, 2)
            y0, x0   = row * hH, col * hW
            mosaic_img[y0:y0 + hH, x0:x0 + hW] = cell

            if len(boxes) > 0:
                b   = boxes.clone()                   # (N, 4) normalized xywh
                # absolute in cell → normalized to full mosaic
                bx  = (b[:, 0] * hW + x0) / W
                by  = (b[:, 1] * hH + y0) / H
                bw  = b[:, 2] * hW / W
                bh  = b[:, 3] * hH / H
                all_boxes.append(torch.stack([bx, by, bw, bh], dim=1))
                all_labels.append(labels)

        if all_boxes:
            boxes_out  = torch.cat(all_boxes,  dim=0)
            labels_out = torch.cat(all_labels, dim=0)
            boxes_out, labels_out = _clip_boxes_xywh(boxes_out, labels_out)
        else:
            boxes_out  = torch.zeros((0, 4), dtype=torch.float32)
            labels_out = torch.zeros((0,),   dtype=torch.int64)

        return mosaic_img, boxes_out, labels_out  # image is already H×W

    # ── geometric augmentations ───────────────────────────────────────────────

    def _random_hflip(self, img, boxes, labels):
        """Horizontal flip with probability hflip_prob."""
        if random.random() > self.hflip_prob:
            return img, boxes, labels
        img = np.fliplr(img).copy()
        if len(boxes) > 0:
            b = boxes.clone()
            b[:, 0] = 1.0 - boxes[:, 0] - boxes[:, 2]   # x = 1 - x - w
            boxes = b
        return img, boxes, labels

    def _random_translate(self, img, boxes, labels):
        """Random translate by up to ±translate_max fraction of image size."""
        if random.random() > self.translate_prob:
            return img, boxes, labels
        H, W = img.shape[:2]
        dx = random.uniform(-self.translate_max, self.translate_max) * W
        dy = random.uniform(-self.translate_max, self.translate_max) * H
        M  = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)

        if len(boxes) > 0:
            b = boxes.clone()
            b[:, 0] = b[:, 0] + dx / W
            b[:, 1] = b[:, 1] + dy / H
            boxes, labels = _clip_boxes_xywh(b, labels)
        return img, boxes, labels

    # ── NEW: RandomZoomOut (shrink into a mean-padded canvas) ─────────────────

    def _random_zoomout(self, img, boxes, labels):
        """Place the image into a (z×) larger mean-filled canvas, then resize back
        to H×W → objects shrink by 1/z. Trains more small-scale instances."""
        H, W = img.shape[:2]
        z = random.uniform(1.0, self.zoomout_max_scale)
        if z <= 1.001:
            return img, boxes, labels
        cH, cW = int(round(H * z)), int(round(W * z))
        canvas = np.empty((cH, cW, 3), dtype=np.uint8)
        canvas[:] = self._fill_rgb
        ox = random.randint(0, cW - W)
        oy = random.randint(0, cH - H)
        canvas[oy:oy + H, ox:ox + W] = img
        out = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)

        if len(boxes) > 0:
            b = boxes.clone()
            b[:, 0] = (ox + boxes[:, 0] * W) / cW
            b[:, 1] = (oy + boxes[:, 1] * H) / cH
            b[:, 2] = boxes[:, 2] * W / cW
            b[:, 3] = boxes[:, 3] * H / cH
            boxes, labels = _clip_boxes_xywh(b, labels)
        return out, boxes, labels

    # ── NEW: SAHI-crop (upper-strip tile up-sampled to IMG_SIZE) ──────────────

    def _sahi_crop(self, img, boxes, labels):
        """Crop a random upper-strip tile (mirroring inference SAHI geometry) and
        up-sample it back to H×W. Matches the magnified distribution SAHI feeds
        the model at inference."""
        H, W = img.shape[:2]
        cols    = max(1, self.sahicrop_cols)
        overlap = min(max(self.sahicrop_overlap, 0.0), 0.9)
        y1 = max(1, int(round(H * self.sahicrop_upper_frac)))
        tw = W / (cols - (cols - 1) * overlap)
        step = tw * (1.0 - overlap)
        i  = random.randint(0, cols - 1)
        x0 = int(round(i * step))
        x1 = min(W, int(round(x0 + tw)))
        if x1 - x0 < 2 or y1 < 2:
            return img, boxes, labels

        crop = img[0:y1, x0:x1]
        out  = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        if len(boxes) > 0:
            cw = float(x1 - x0)
            ch = float(y1)
            bx1 = (boxes[:, 0] * W - x0) / cw
            by1 = (boxes[:, 1] * H) / ch
            bw  = (boxes[:, 2] * W) / cw
            bh  = (boxes[:, 3] * H) / ch
            b = torch.stack([bx1, by1, bw, bh], dim=1)
            boxes, labels = _clip_boxes_xywh(b, labels, min_vis=self.sahicrop_min_vis)
        else:
            boxes, labels = boxes, labels
        return out, boxes, labels

    # ── NEW: copy-paste rare-class instances ──────────────────────────────────

    def _copy_paste(self, img, boxes, labels):
        """Paste a few extra target-class instances (rectangular crops from the
        source pool, scale-jittered) into the current H×W image."""
        if not self._cp_pool:
            return img, boxes, labels
        H, W = img.shape[:2]
        n = random.randint(1, max(1, self.copypaste_max_per_img))
        new_boxes, new_labels = [], []

        for _ in range(n):
            path, (sx, sy, sw, sh), lab, src_w, src_h = random.choice(self._cp_pool)
            src = cv2.imread(path, cv2.IMREAD_COLOR)
            if src is None:
                continue
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            sh_img, sw_img = src.shape[:2]
            # guard against annotation/image size drift
            x0 = int(max(0, min(sw_img - 1, round(sx))))
            y0 = int(max(0, min(sh_img - 1, round(sy))))
            x1 = int(max(x0 + 1, min(sw_img, round(sx + sw))))
            y1 = int(max(y0 + 1, min(sh_img, round(sy + sh))))
            patch = src[y0:y1, x0:x1]
            if patch.size == 0:
                continue

            # target paste size: keep object's relative scale, then jitter
            s = random.uniform(*self.copypaste_scale_range)
            pw = int(round((sw / src_w) * W * s))
            ph = int(round((sh / src_h) * H * s))
            if pw < 3 or ph < 3 or pw >= W or ph >= H:
                continue
            patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

            # placement: lights biased to upper region, others to lower region
            name = self.class_names[lab]
            if name in self._cp_upper:
                py = random.randint(0, max(0, int(H * 0.55) - ph))
            else:
                py = random.randint(min(int(H * 0.40), H - ph), H - ph)
            px = random.randint(0, W - pw)

            img[py:py + ph, px:px + pw] = patch

            # Remove existing boxes heavily occluded by the pasted patch
            if boxes.numel() > 0:
                ex1, ey1 = boxes[:, 0], boxes[:, 1]
                ex2, ey2 = ex1 + boxes[:, 2], ey1 + boxes[:, 3]
                px1n, py1n = px / W, py / H
                px2n, py2n = (px + pw) / W, (py + ph) / H
                inter_w = (ex2.clamp(max=px2n) - ex1.clamp(min=px1n)).clamp(min=0)
                inter_h = (ey2.clamp(max=py2n) - ey1.clamp(min=py1n)).clamp(min=0)
                occluded_frac = (inter_w * inter_h) / (boxes[:, 2] * boxes[:, 3]).clamp(min=1e-6)
                keep = occluded_frac < 0.5
                boxes = boxes[keep]
                labels = labels[keep]

            new_boxes.append([px / W, py / H, pw / W, ph / H])
            new_labels.append(lab)

        if new_boxes:
            nb = torch.as_tensor(new_boxes, dtype=torch.float32)
            nl = torch.as_tensor(new_labels, dtype=torch.int64)
            boxes  = torch.cat([nb, boxes], dim=0) if boxes.numel() else nb
            labels = torch.cat([nl, labels], dim=0) if labels.numel() else nl
        return img, boxes, labels

    # ── photometric augmentation (RGB→HSV) ────────────────────────────────────

    def photometric_augment(self, img,
                            brightness_delta=32,
                            contrast_range=(0.8, 1.2),
                            saturation_range=None,
                            hue_delta=None,
                            p_jitter=0.5,
                            p_blur=0.3,
                            p_noise=0.3):
        """
        img: H×W×3 RGB uint8 → same shape RGB uint8.
        FIX vs v2: cv2.COLOR_BGR2HSV → cv2.COLOR_RGB2HSV (and HSV2RGB).
        """
        out = img.astype(np.float32)
        if hue_delta is None:
            hue_delta = self.hue_delta
        if saturation_range is None:
            saturation_range = self.saturation_range

        if random.random() < p_jitter:
            out += random.uniform(-brightness_delta, brightness_delta)

        if random.random() < p_jitter:
            out *= random.uniform(*contrast_range)

        sat_is_noop = (abs(saturation_range[0] - 1.0) < 1e-6 and
                       abs(saturation_range[1] - 1.0) < 1e-6)
        if random.random() < p_jitter and not (hue_delta <= 0 and sat_is_noop):
            hsv = cv2.cvtColor(out.clip(0, 255).astype(np.uint8),
                               cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(*saturation_range)
            if hue_delta > 0:
                hsv[..., 0] += random.uniform(-hue_delta, hue_delta)
            out = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8),
                               cv2.COLOR_HSV2RGB).astype(np.float32)

        if random.random() < p_blur:
            k   = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)

        if random.random() < p_noise:
            out += np.random.randn(*out.shape) * random.uniform(5, 20)

        return out.clip(0, 255).astype(np.uint8)

    # ── main getitem ──────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        if self.use_mosaic and np.random.rand() < self.mosaic_prob:
            image, boxes, labels = self._mosaic(idx)
            # mosaic already at H×W — no resize needed
        else:
            image, boxes, labels = self._load_raw(idx)
            image = self._resize_rgb(image)

        if self.mode == 'train':
            # scale aug FIRST: zoom-out (more small objs) XOR sahi-crop (match inference).
            # Mutually exclusive so we never both shrink and magnify the same frame.
            r = random.random()
            if r < self.zoomout_prob:
                image, boxes, labels = self._random_zoomout(image, boxes, labels)
            elif r < self.zoomout_prob + self.sahicrop_prob:
                image, boxes, labels = self._sahi_crop(image, boxes, labels)

            # THEN copy-paste — pasted instances land in the visible region
            if self._cp_pool and random.random() < self.copypaste_prob:
                image, boxes, labels = self._copy_paste(image, boxes, labels)

            # photometric
            if self.augment_prob > 0 and np.random.rand() < self.augment_prob:
                image = self.photometric_augment(image)

            # existing geometric
            image, boxes, labels = self._random_hflip(image, boxes, labels)
            image, boxes, labels = self._random_translate(image, boxes, labels)

        return self.image_to_tensor(image), boxes, labels
