import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import matplotlib.pyplot as plt

def decode_outputs(
    model_outputs,   # dict with lists per-level like in your model
    image_shape,     # (H_img, W_img)
    strides = [16, 32, 64],
    score_thresh = 0.05,
    nms_thresh = 0.6,
    max_detections = 100,
    return_angles = False,   # if True, also returns OBB angles tensor (K,)
):
    """
    Decode FCOS-style model outputs (reg outputs are LTRB pixel offsets, >=0).

    Returns
    -------
    boxes  : (K, 4) xyxy
    scores : (K,)
    labels : (K,)   int64, 0..C-1
    angles : (K,)   float, OBB θ in rad — only when return_angles=True AND
                    'angle' key is present in model_outputs.

    Notes
    -----
    Assumes batch size 1 at inference.
    strides must correspond to each FPN level in model_outputs['cls'].
    """
    device = model_outputs['cls'][0].device
    num_levels = len(model_outputs['cls'])
    has_angle = return_angles and ('angle' in model_outputs)

    all_boxes  = []
    all_scores = []
    all_labels = []
    all_angles = []   # populated only when has_angle

    for lvl_idx in range(num_levels):
        cls_logits = model_outputs['cls'][lvl_idx]  # (B,C,H,W)
        reg_out    = model_outputs['reg'][lvl_idx]  # (B,4,H,W)
        ctr_logits = model_outputs['ctr'][lvl_idx]  # (B,1,H,W)

        assert cls_logits.shape[0] == 1, "decode_outputs assumes batch size 1 at inference."
        cls_logits = cls_logits[0]   # (C,H,W)
        reg_out    = reg_out[0]      # (4,H,W)
        ctr_logits = ctr_logits[0]   # (1,H,W)

        C, H, W = cls_logits.shape
        stride = strides[lvl_idx]

        shifts_x = (torch.arange(0, W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(0, H, device=device) + 0.5) * stride
        ys, xs   = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        centers  = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)  # (Nloc,2)

        cls_logits_flat = cls_logits.reshape(C, -1).permute(1, 0)  # (Nloc,C)
        reg_flat        = reg_out.reshape(4, -1).permute(1, 0)     # (Nloc,4)
        ctr_logits_flat = ctr_logits.reshape(-1)                   # (Nloc,)

        cls_prob    = torch.sigmoid(cls_logits_flat)                  # (Nloc,C)
        ctr_prob    = torch.sigmoid(ctr_logits_flat).unsqueeze(1)     # (Nloc,1)
        final_scores = cls_prob * ctr_prob                            # (Nloc,C)

        # pre-filter locations where at least one class clears the threshold
        if isinstance(score_thresh, (list, tuple)):
            thresh_tensor = torch.tensor(score_thresh, device=device, dtype=final_scores.dtype)
            if thresh_tensor.numel() != C:
                raise ValueError(f"score_thresh has {thresh_tensor.numel()} elements but model has {C} classes")
            keep_mask_any = (final_scores > thresh_tensor.unsqueeze(0)).any(dim=1)
        elif torch.is_tensor(score_thresh) and score_thresh.numel() > 1:
            thresh_tensor = score_thresh.to(device=device, dtype=final_scores.dtype)
            if thresh_tensor.numel() != C:
                raise ValueError(f"score_thresh has {thresh_tensor.numel()} elements but model has {C} classes")
            keep_mask_any = (final_scores > thresh_tensor.unsqueeze(0)).any(dim=1)
        else:
            keep_mask_any = (final_scores > float(score_thresh)).any(dim=1)

        if keep_mask_any.sum() == 0:
            continue

        centers_keep = centers[keep_mask_any]                  # (Nk,2)
        reg_keep     = reg_flat[keep_mask_any].clamp(min=0.0)  # (Nk,4)
        scores_keep  = final_scores[keep_mask_any]             # (Nk,C)

        Nk = scores_keep.shape[0]
        cls_idx_grid = torch.arange(C, device=device).unsqueeze(0).expand(Nk, C)  # (Nk,C)

        x_cent = centers_keep[:, 0].unsqueeze(1).expand(-1, C)
        y_cent = centers_keep[:, 1].unsqueeze(1).expand(-1, C)
        l_ = reg_keep[:, 0].unsqueeze(1).expand(-1, C)
        t_ = reg_keep[:, 1].unsqueeze(1).expand(-1, C)
        r_ = reg_keep[:, 2].unsqueeze(1).expand(-1, C)
        b_ = reg_keep[:, 3].unsqueeze(1).expand(-1, C)

        boxes_lvl  = torch.stack([x_cent - l_, y_cent - t_, x_cent + r_, y_cent + b_], dim=2).reshape(-1, 4)
        scores_lvl = scores_keep.reshape(-1)
        labels_lvl = cls_idx_grid.reshape(-1)

        # angle: (Nk,) → expand per-class → (Nk*C,)
        if has_angle:
            ang_lvl = model_outputs['angle'][lvl_idx][0].reshape(-1)[keep_mask_any]  # (Nk,)
            ang_lvl = ang_lvl.unsqueeze(1).expand(-1, C).reshape(-1)                # (Nk*C,)

        # per-class score threshold filter
        if isinstance(score_thresh, (list, tuple)):
            thresh_tensor = torch.tensor(score_thresh, device=device, dtype=scores_lvl.dtype)
            keep = scores_lvl > thresh_tensor[labels_lvl]
        elif torch.is_tensor(score_thresh) and score_thresh.numel() > 1:
            keep = scores_lvl > score_thresh.to(device)[labels_lvl]
        else:
            keep = scores_lvl > float(score_thresh)

        if keep.sum() == 0:
            continue

        all_boxes.append(boxes_lvl[keep])
        all_scores.append(scores_lvl[keep])
        all_labels.append(labels_lvl[keep])
        if has_angle:
            all_angles.append(ang_lvl[keep])

    # ---- no detections ----
    empty4 = torch.empty((0, 4), device=device)
    empty1 = torch.empty((0,),   device=device)
    empty_long = torch.empty((0,), dtype=torch.long, device=device)
    if len(all_boxes) == 0:
        if return_angles:
            return empty4, empty1, empty_long, empty1
        return empty4, empty1, empty_long

    boxes  = torch.cat(all_boxes,  dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)
    if has_angle:
        angles = torch.cat(all_angles, dim=0)

    # clip to image
    Himg, Wimg = image_shape
    boxes[:, 0].clamp_(0.0, float(Wimg))
    boxes[:, 1].clamp_(0.0, float(Himg))
    boxes[:, 2].clamp_(0.0, float(Wimg))
    boxes[:, 3].clamp_(0.0, float(Himg))

    # per-class NMS
    keep_indices = []
    for lab in labels.unique():
        mask     = (labels == lab)
        boxes_l  = boxes[mask]
        scores_l = scores[mask]
        if boxes_l.numel() == 0:
            continue
        keep_l = nms(boxes_l, scores_l, nms_thresh)
        if keep_l.numel() == 0:
            continue
        global_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)[keep_l]
        keep_indices.append(global_idx)

    if len(keep_indices) == 0:
        if return_angles:
            return empty4, empty1, empty_long, empty1
        return empty4, empty1, empty_long

    keep_indices = torch.cat(keep_indices)
    sorted_idx   = torch.argsort(scores[keep_indices], descending=True)
    sorted_idx   = keep_indices[sorted_idx][:max_detections]

    if return_angles:
        ang_out = angles[sorted_idx] if has_angle else torch.zeros(sorted_idx.shape[0], device=device)
        return boxes[sorted_idx], scores[sorted_idx], labels[sorted_idx], ang_out
    return boxes[sorted_idx], scores[sorted_idx], labels[sorted_idx]

def generate_detection_overlay(image_np, boxes_xyxy, scores, labels, class_names=None):
    """
    Return image with detections drawn on it (numpy array).
    image_np: HxWx3 numpy (uint8 or float).
    boxes_xyxy: tensor (K,4) xyxy in pixels (can be on GPU or CPU).
    """

    # move to cpu numpy
    boxes_np = boxes_xyxy.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    # convert image to uint8 if float in [0,1]
    if image_np.dtype.kind == 'f':
        img = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = image_np.astype(np.uint8)

    img_out = img.copy()

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = map(int, box.tolist())
        score = float(scores_np[i])
        lab = int(labels_np[i])
        label_text = f"{lab}:{score:.2f}" if class_names is None else f"{class_names[lab]}:{score:.2f}"

        # Draw rectangle
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box

        # Put text above box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 255), -1)  # filled background
        cv2.putText(img_out, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_out


def detection_inference(model_detection, feats, img_size, score_thresh=0.2, nms_thresh=0.6,
                        per_class_thresh=None, return_angles=False):
    """Run detection inference on pre-extracted backbone features.

    Parameters
    ----------
    model_detection  : detection head (DinoFCOSHead)
    feats            : backbone feature map (B=1, C, H, W)
    img_size         : (H, W) of the input image
    score_thresh     : scalar confidence threshold fallback
    nms_thresh       : IoU threshold for NMS
    per_class_thresh : optional list/tensor of per-class thresholds
    return_angles    : if True, also returns OBB angle tensor (K,)

    Returns
    -------
    boxes, scores, labels  — always
    angles                 — additionally when return_angles=True
    """
    outputs = model_detection(feats)

    img_h, img_w  = img_size
    feat_w        = outputs['cls'][0].shape[3]
    first_stride  = float(img_w / feat_w)
    strides       = [first_stride * (2 ** l) for l in range(len(outputs['cls']))]

    effective_thresh = per_class_thresh if per_class_thresh is not None else score_thresh

    return decode_outputs(
        outputs,
        img_size,
        strides,
        score_thresh=effective_thresh,
        nms_thresh=nms_thresh,
        return_angles=return_angles,
    )


def plot_detections(image_np, boxes_xyxy, scores, labels, class_names=None, figsize=(10,10)):
    """
    Plot detections. image_np: HxWx3 numpy (uint8 or float).
    boxes_xyxy: tensor (K,4) xyxy in pixels (can be on GPU or CPU).
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)
    # convert image to uint8 if float in [0,1]
    img = generate_detection_overlay(image_np, boxes_xyxy, scores, labels, class_names)
    ax.imshow(img)

    ax.axis('off')
    plt.show()

# ===========================================================================
# OBB utilities (merged from utils_OBB.py)
# ===========================================================================

import math


# ---------------------------------------------------------------------------
# Probabilistic IoU helpers
# ---------------------------------------------------------------------------

def _get_covariance_matrix(
    boxes_xywhr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 2-D Gaussian covariance components from (cx,cy,w,h,θ) boxes.

    Each rotated box is modelled as a uniform distribution over its interior;
    the covariance of a uniform distribution over a rectangle of width w,
    height h is diag(w²/12, h²/12) in the box frame, rotated to world frame.

    Parameters
    ----------
    boxes_xywhr : (..., 5) — (cx, cy, w, h, θ) in pixels and radians

    Returns
    -------
    a, b, c : (...,) scalars representing the 2×2 covariance matrix
              Σ = [[a, c],
                   [c, b]]
    """
    w2  = boxes_xywhr[..., 2].pow(2) / 12.0
    h2  = boxes_xywhr[..., 3].pow(2) / 12.0
    cos = boxes_xywhr[..., 4].cos()
    sin = boxes_xywhr[..., 4].sin()
    a   = cos.pow(2) * w2 + sin.pow(2) * h2   # Σ[0,0]
    b   = sin.pow(2) * w2 + cos.pow(2) * h2   # Σ[1,1]
    c   = (w2 - h2) * cos * sin               # Σ[0,1]
    return a, b, c


def probiou(
    obb1: torch.Tensor,
    obb2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Element-wise Probabilistic IoU between two matched sets of OBBs.

    Computes the Hellinger distance between the Gaussian distributions
    of each box pair and converts to an IoU-like overlap score in [0, 1].

    Parameters
    ----------
    obb1, obb2 : (N, 5) — (cx, cy, w, h, θ)

    Returns
    -------
    (N,) tensor of ProBIoU values ∈ [0, 1]  (higher = more overlap)
    """
    x1, y1 = obb1[:, 0], obb1[:, 1]
    x2, y2 = obb2[:, 0], obb2[:, 1]
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    # Bhattacharyya distance (3 terms)
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / denom * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) /
        (4.0 * ((a1 * b1 - c1.pow(2)).clamp(min=0) *
                (a2 * b2 - c2.pow(2)).clamp(min=0)).sqrt() + eps) + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()   # Hellinger distance ∈ [0, 1]
    return 1.0 - hd                          # ProBIoU ∈ [0, 1]


def batch_probiou(
    obb1: torch.Tensor,
    obb2: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """All-pairs Probabilistic IoU between two sets of OBBs.

    Parameters
    ----------
    obb1 : (N, 5) — (cx, cy, w, h, θ)
    obb2 : (M, 5) — (cx, cy, w, h, θ)

    Returns
    -------
    (N, M) ProBIoU matrix ∈ [0, 1]
    """
    x1 = obb1[:, 0:1]            # (N, 1)
    y1 = obb1[:, 1:2]
    x2 = obb2[:, 0:1].T          # (1, M)
    y2 = obb2[:, 1:2].T

    a1, b1, c1 = _get_covariance_matrix(obb1)   # (N,)
    a2, b2, c2 = _get_covariance_matrix(obb2)   # (M,)

    # Broadcast to (N, M)
    a1 = a1[:, None]; b1 = b1[:, None]; c1 = c1[:, None]
    a2 = a2[None, :]; b2 = b2[None, :]; c2 = c2[None, :]

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / denom * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) /
        (4.0 * ((a1 * b1 - c1.pow(2)).clamp(min=0) *
                (a2 * b2 - c2.pow(2)).clamp(min=0)).sqrt() + eps) + eps
    ).log() * 0.5

    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1.0 - hd   # (N, M)


# ---------------------------------------------------------------------------
# OBB geometry helpers
# ---------------------------------------------------------------------------

def dist2rbox(
    anc_points: torch.Tensor,
    ltrb: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    """Convert FCOS LTRB predictions + rotation angle to (cx, cy, w, h, θ).

    Parameters
    ----------
    anc_points : (N, 2)  anchor center coordinates (cx, cy) in pixels
    ltrb       : (N, 4)  predicted distances [l, t, r, b], all ≥ 0
    angles     : (N,)    predicted OBB angle θ in radians

    Returns
    -------
    (N, 5) tensor: (cx, cy, w, h, θ)
    """
    x1 = anc_points[:, 0] - ltrb[:, 0]
    y1 = anc_points[:, 1] - ltrb[:, 1]
    x2 = anc_points[:, 0] + ltrb[:, 2]
    y2 = anc_points[:, 1] + ltrb[:, 3]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1).clamp(min=0.0)
    h  = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h, angles], dim=-1)


def xywhr2xyxyxyxy(boxes_xywhr: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h, θ) OBBs to 4-corner representation.

    Corner order (before rotation, CCW from top-left):
        0: top-left  (-w/2, -h/2)
        1: top-right (+w/2, -h/2)
        2: bot-right (+w/2, +h/2)
        3: bot-left  (-w/2, +h/2)

    Parameters
    ----------
    boxes_xywhr : (N, 5) — (cx, cy, w, h, θ)

    Returns
    -------
    (N, 4, 2) corner coordinates in image pixel space
    """
    cx  = boxes_xywhr[:, 0]
    cy  = boxes_xywhr[:, 1]
    hw  = boxes_xywhr[:, 2] * 0.5
    hh  = boxes_xywhr[:, 3] * 0.5
    cos = boxes_xywhr[:, 4].cos()
    sin = boxes_xywhr[:, 4].sin()

    # Unrotated corner offsets from center: (N, 4)
    dx = torch.stack([-hw,  hw,  hw, -hw], dim=1)
    dy = torch.stack([-hh, -hh,  hh,  hh], dim=1)

    # Apply rotation matrix [[cos, −sin], [sin, cos]]
    rx = cos[:, None] * dx - sin[:, None] * dy + cx[:, None]
    ry = sin[:, None] * dx + cos[:, None] * dy + cy[:, None]

    return torch.stack([rx, ry], dim=-1)   # (N, 4, 2)


# ---------------------------------------------------------------------------
# Rotated NMS
# ---------------------------------------------------------------------------

def rotated_nms_per_class(
    boxes_xywhr: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.6,
    max_detections: int = 300,
) -> torch.Tensor:
    """Per-class greedy NMS using ProBIoU overlap for rotated boxes.

    Parameters
    ----------
    boxes_xywhr   : (N, 5) — (cx, cy, w, h, θ)
    scores        : (N,)
    labels        : (N,) int64 class indices
    iou_threshold : suppression threshold (same scale as ProBIoU ∈ [0,1])
    max_detections: total cap on returned detections

    Returns
    -------
    keep : 1-D LongTensor of kept global indices sorted by score (descending)
    """
    device = boxes_xywhr.device
    if boxes_xywhr.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    keep_all = []
    for lab in labels.unique():
        mask = labels == lab
        b    = boxes_xywhr[mask]
        s    = scores[mask]
        gidx = mask.nonzero(as_tuple=False).squeeze(1)

        order = s.argsort(descending=True)
        b = b[order]
        gidx = gidx[order]

        survived = torch.ones(b.shape[0], dtype=torch.bool, device=device)
        for i in range(b.shape[0]):
            if not survived[i]:
                continue
            keep_all.append(gidx[i].item())
            if i + 1 >= b.shape[0]:
                break
            # Suppress all remaining boxes with ProBIoU > threshold
            remaining = survived[i + 1:].nonzero(as_tuple=False).squeeze(1)
            if remaining.numel() == 0:
                break
            iou = batch_probiou(b[i:i + 1], b[i + 1:][remaining])[0]  # (M,)
            suppress = iou > iou_threshold
            survived_idx = (i + 1) + remaining[suppress]
            survived[survived_idx] = False

    if not keep_all:
        return torch.empty(0, dtype=torch.long, device=device)

    keep_t = torch.tensor(keep_all, dtype=torch.long, device=device)
    sorted_idx = scores[keep_t].argsort(descending=True)
    return keep_t[sorted_idx[:max_detections]]


# ---------------------------------------------------------------------------
# OBB decode pipeline
# ---------------------------------------------------------------------------

def _suppress_group_crossclass(boxes_xywhr, scores, labels, group_labels, iou_threshold):
    """Class-agnostic greedy ProBIoU NMS *within* a set of class labels.

    Boxes whose label is NOT in ``group_labels`` are always kept. Among boxes whose
    label IS in the group, overlapping boxes (ProBIoU > iou_threshold) are suppressed
    regardless of their (differing) class — only the highest-scoring one survives.
    Used to collapse the three traffic_light_* colour classes so a single physical
    light yields one box at its top-scoring colour. Returns a bool keep-mask.
    """
    n = boxes_xywhr.shape[0]
    keep = torch.ones(n, dtype=torch.bool, device=boxes_xywhr.device)
    if n <= 1 or not group_labels:
        return keep
    in_group = torch.tensor([int(l) in group_labels for l in labels.tolist()],
                            dtype=torch.bool, device=boxes_xywhr.device)
    g_idx = torch.nonzero(in_group, as_tuple=False).flatten()
    if g_idx.numel() <= 1:
        return keep
    order = g_idx[torch.argsort(scores[g_idx], descending=True)]   # abs idx, score desc
    b = boxes_xywhr[order]
    m = order.numel()
    dead = torch.zeros(m, dtype=torch.bool, device=boxes_xywhr.device)
    for i in range(m):
        if dead[i]:
            continue
        if i + 1 < m:
            iou = batch_probiou(b[i:i + 1], b[i + 1:])[0]          # (m-i-1,)
            dead[i + 1:] |= (iou > iou_threshold)
    keep[order[dead]] = False
    return keep


def decode_outputs_OBB(
    model_outputs,
    image_shape: tuple[int, int],
    strides: list[float] | None = None,
    score_thresh=0.05,
    nms_thresh: float = 0.6,
    max_detections: int = 300,
    tl_group_labels=None,
    use_centerness: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode FCOS-OBB model outputs to rotated bounding boxes.

    Parameters
    ----------
    model_outputs : dict — 'cls', 'reg', 'ctr', 'angle' lists (per FPN level)
                          Assumes batch size 1 at inference.
    image_shape   : (H, W) of the input image
    strides       : per-level strides; inferred from feature map sizes if None
    score_thresh  : scalar or per-class list confidence threshold
    nms_thresh    : ProBIoU threshold for rotated NMS
    max_detections: maximum returned detections

    Returns
    -------
    boxes  : (K, 5) — (cx, cy, w, h, θ) in pixels / radians
    scores : (K,)
    labels : (K,) int64 class indices (0-indexed)
    """
    device = model_outputs['cls'][0].device
    num_levels = len(model_outputs['cls'])
    # Heads trained with obb=False emit no 'angle' key — treat all angles as 0
    # (axis-aligned). dist2rbox then produces θ=0 boxes equivalent to AABB.
    has_angle = 'angle' in model_outputs

    if strides is None:
        img_h, img_w = image_shape
        feat_w = model_outputs['cls'][0].shape[3]
        s0 = float(img_w / feat_w)
        strides = [s0 * (2 ** lvl) for lvl in range(num_levels)]

    all_boxes  = []
    all_scores = []
    all_labels = []

    for lvl in range(num_levels):
        cls_logits = model_outputs['cls'][lvl][0]    # (C, H, W)
        reg_out    = model_outputs['reg'][lvl][0]    # (4, H, W)
        ctr_logits = model_outputs['ctr'][lvl][0]    # (1, H, W)
        ang_out    = model_outputs['angle'][lvl][0] if has_angle else None  # (1, H, W)

        C, H, W = cls_logits.shape
        stride = strides[lvl]

        shifts_x = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
        ys, xs   = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        centers  = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1)   # (Nloc, 2)

        cls_flat = cls_logits.reshape(C, -1).permute(1, 0)   # (Nloc, C)

        # Handle DFL regression: 4*(reg_max+1) channels → decode to 4
        reg_ch = reg_out.shape[0]
        if reg_ch > 4:
            reg_max = reg_ch // 4 - 1
            reg_reshaped = reg_out.reshape(4, reg_max + 1, -1)  # (4, rm+1, Nloc)
            proj = torch.arange(reg_max + 1, device=device, dtype=torch.float32)
            reg_flat = (F.softmax(reg_reshaped, dim=1) *
                        proj.reshape(1, -1, 1)).sum(dim=1)      # (4, Nloc)
            reg_flat = reg_flat.permute(1, 0)                    # (Nloc, 4)
        else:
            reg_flat = reg_out.reshape(4, -1).permute(1, 0)      # (Nloc, 4)

        ctr_flat = ctr_logits.reshape(-1)                      # (Nloc,)
        ang_flat = (ang_out.reshape(-1) if has_angle
                    else torch.zeros(H * W, device=device, dtype=torch.float32))  # (Nloc,)

        cls_prob = torch.sigmoid(cls_flat)
        if use_centerness:
            ctr_prob   = torch.sigmoid(ctr_flat).unsqueeze(1)
            scores_all = cls_prob * ctr_prob                   # (Nloc, C)
        else:
            scores_all = cls_prob                               # VFL scores are IoU-aware

        # Pre-filter locations where at least one class clears threshold
        if isinstance(score_thresh, (list, tuple)):
            thresh_t  = torch.tensor(score_thresh, device=device, dtype=scores_all.dtype)
            keep_mask = (scores_all > thresh_t.unsqueeze(0)).any(dim=1)
        else:
            keep_mask = (scores_all > float(score_thresh)).any(dim=1)

        if keep_mask.sum() == 0:
            continue

        centers_k = centers[keep_mask]
        reg_k     = reg_flat[keep_mask].clamp(min=0.0)
        scores_k  = scores_all[keep_mask]                      # (Nk, C)
        ang_k     = ang_flat[keep_mask]                        # (Nk,)

        Nk = scores_k.shape[0]
        cls_idx = torch.arange(C, device=device).unsqueeze(0).expand(Nk, -1)  # (Nk, C)

        # Decode OBB; expand per class
        boxes_xywhr = dist2rbox(centers_k, reg_k, ang_k)   # (Nk, 5)
        boxes_exp   = boxes_xywhr.unsqueeze(1).expand(-1, C, -1).reshape(-1, 5)  # (Nk*C, 5)
        scores_exp  = scores_k.reshape(-1)
        labels_exp  = cls_idx.reshape(-1)

        # Per-class score threshold
        if isinstance(score_thresh, (list, tuple)):
            thresh_t = torch.tensor(score_thresh, device=device, dtype=scores_exp.dtype)
            keep = scores_exp > thresh_t[labels_exp]
        else:
            keep = scores_exp > float(score_thresh)

        if keep.sum() == 0:
            continue

        all_boxes.append(boxes_exp[keep])
        all_scores.append(scores_exp[keep])
        all_labels.append(labels_exp[keep])

    # ---- no detections ----
    empty5    = torch.empty((0, 5), device=device)
    empty1    = torch.empty((0,),   device=device)
    empty_int = torch.empty((0,), dtype=torch.long, device=device)
    if not all_boxes:
        return empty5, empty1, empty_int

    boxes  = torch.cat(all_boxes,  dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Clip centres to image bounds
    img_h, img_w = image_shape
    boxes[:, 0].clamp_(0.0, float(img_w))
    boxes[:, 1].clamp_(0.0, float(img_h))

    # Rotated NMS per class
    keep = rotated_nms_per_class(boxes, scores, labels, nms_thresh, max_detections)

    if keep.numel() == 0:
        return empty5, empty1, empty_int

    boxes_k, scores_k, labels_k = boxes[keep], scores[keep], labels[keep]

    # Fix A: collapse cross-class duplicates within a label group (the three
    # traffic_light_* classes). Per-class NMS keeps one box PER colour, so one light
    # can show red+green+other boxes; this keeps only the top-scoring colour per light.
    if tl_group_labels:
        sub = _suppress_group_crossclass(
            boxes_k, scores_k, labels_k,
            set(int(x) for x in tl_group_labels), nms_thresh)
        boxes_k, scores_k, labels_k = boxes_k[sub], scores_k[sub], labels_k[sub]

    return boxes_k, scores_k, labels_k


def detection_inference_OBB(
    model_detection,
    feats: torch.Tensor,
    img_size: tuple[int, int],
    score_thresh: float = 0.2,
    nms_thresh: float = 0.6,
    per_class_thresh=None,
    max_detections: int = 300,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run OBB detection inference on pre-extracted backbone features.

    Parameters
    ----------
    model_detection : DinoFCOSHeadOBB instance
    feats           : (1, C, H, W) backbone feature map
    img_size        : (H, W) of the input image (before backbone stride)
    score_thresh    : scalar confidence threshold fallback
    nms_thresh      : ProBIoU threshold for rotated NMS
    per_class_thresh: list of per-class thresholds (overrides score_thresh)
    max_detections  : maximum returned detections

    Returns
    -------
    boxes_xywhr : (K, 5) — (cx, cy, w, h, θ)
    scores      : (K,)
    labels      : (K,) int64
    """
    outputs = model_detection(feats)

    img_h, img_w = img_size
    feat_w  = outputs['cls'][0].shape[3]
    s0      = float(img_w / feat_w)
    strides = [s0 * (2 ** lvl) for lvl in range(len(outputs['cls']))]

    effective_thresh = per_class_thresh if per_class_thresh is not None else score_thresh

    return decode_outputs_OBB(
        outputs,
        img_size,
        strides=strides,
        score_thresh=effective_thresh,
        nms_thresh=nms_thresh,
        max_detections=max_detections,
    )
