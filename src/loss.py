"""
Enhanced TAL/STAL loss for DINOv3 + FCOS-OBB head.

Fixes vs. loss_new.py
----------------------
1. Angle loss  : sin(2·Δθ)² × aspect_weight
   Previous smooth_L1 fails at boundary (±π/4) because small angular errors
   near the boundary wrap to large loss values.  sin(2·Δθ)² is π/2-periodic,
   correctly handles wrap-around, and downweights square objects (aspect≈1)
   where angle prediction is irrelevant.

2. STAL IoU    : ProBIoU (Probabilistic IoU for OBBs) replaces CIoU in the
   TAL alignment metric when angle predictions are available.  ProBIoU is
   rotation-aware so the assignment better matches actual OBB overlap.

3. STAL loops  : _topk_mask() fully vectorised — no Python for-loops.
   Previous O(bsz × num_gt) loop blocked CUDA stream.

4. CUDA OOM    : __call__ wraps the expensive STAL step in try/except and
   falls back to CPU computation, then moves results back to GPU.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.decode import (
    probiou,
    batch_probiou,
    dist2rbox,
    _get_covariance_matrix,
)


# ---------------------------------------------------------------------------
# Shared IoU helpers (kept from loss_new.py for non-OBB fallback)
# ---------------------------------------------------------------------------

def bbox_iou_ciou(
    box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou   = inter / union
    cw    = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch    = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2    = cw ** 2 + ch ** 2 + eps
    rho2  = (((b1_x1 + b1_x2) - (b2_x1 + b2_x2)) ** 2 +
             ((b1_y1 + b1_y2) - (b2_y1 + b2_y2)) ** 2) / 4.0
    v     = (4.0 / (math.pi ** 2)) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
    ) ** 2
    with torch.no_grad():
        alpha_ciou = (v / (v - iou + 1.0 + eps)).clamp(0.0, 10.0)
    return iou - (rho2 / c2 + v * alpha_ciou)


def ciou_loss(
    pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    return (1.0 - bbox_iou_ciou(pred_boxes, target_boxes, eps).clamp(-1.0, 1.0)).sum()


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------

def make_anchors(
    feats: List[torch.Tensor],
    strides: List[float],
    offset: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    anc_pts, stride_vals = [], []
    device = feats[0].device
    dtype  = feats[0].dtype
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = (torch.arange(w, device=device, dtype=dtype) + offset) * stride
        sy = (torch.arange(h, device=device, dtype=dtype) + offset) * stride
        gy, gx = torch.meshgrid(sy, sx, indexing='ij')
        anc_pts.append(torch.stack([gx.flatten(), gy.flatten()], dim=-1))
        stride_vals.append(torch.full((h * w, 1), stride, device=device, dtype=dtype))
    return torch.cat(anc_pts, dim=0), torch.cat(stride_vals, dim=0)


# ---------------------------------------------------------------------------
# STAL: Small-Target-Aware Label Assignment (OBB version)
# ---------------------------------------------------------------------------

class STALTaskAlignedAssignerOBB:
    """STAL assigner with ProBIoU alignment, vectorised topk, and OOM guard.

    Drop-in replacement for STALTaskAlignedAssigner in loss_new.py.

    When ``pd_angles`` is passed to __call__(), the TAL alignment metric uses
    ProBIoU (rotation-aware) instead of CIoU.  Without pd_angles it falls
    back to CIoU for backwards compatibility.
    """

    def __init__(
        self,
        topk: int = 10,
        num_classes: int = 80,
        alpha: float = 0.5,
        beta: float = 4.0,
        eps: float = 1e-9,
        min_stride: float = 8.0,
        stal_stride: float = 16.0,
        min_candidates: int = 8,
    ):
        self.topk           = topk
        self.num_classes    = num_classes
        self.alpha          = alpha
        self.beta           = beta
        self.eps            = eps
        self.min_stride     = min_stride
        self.stal_stride    = stal_stride
        self.min_candidates = min_candidates

    # ---- size buckets -------------------------------------------------------

    def _size_buckets(self, short_side: torch.Tensor):
        tiny  = short_side <= (2.0 * self.min_stride)
        small = (short_side > (2.0 * self.min_stride)) & \
                (short_side <= (4.0 * self.min_stride))
        return tiny, small

    def _adaptive_expand(self, gt_w, gt_h):
        short_side = torch.minimum(gt_w, gt_h)
        tiny, small = self._size_buckets(short_side)
        expand = torch.full_like(short_side, self.stal_stride)
        expand = torch.where(small, torch.full_like(expand, 3.0 * self.min_stride), expand)
        expand = torch.where(tiny,  torch.full_like(expand, 4.0 * self.min_stride), expand)
        return expand, tiny, small

    def _adaptive_topk(self, short_side: torch.Tensor):
        tiny, small = self._size_buckets(short_side)
        topk = torch.full_like(short_side, float(self.topk))
        topk = torch.where(small, torch.full_like(topk, float(max(self.topk + 4, 14))), topk)
        topk = torch.where(tiny,  torch.full_like(topk, float(max(self.topk + 8, 18))), topk)
        return topk.long(), tiny, small

    def _adaptive_beta(self, short_side: torch.Tensor):
        tiny, small = self._size_buckets(short_side)
        beta = torch.full_like(short_side, self.beta)
        beta = torch.where(small, torch.full_like(beta, 3.0), beta)
        beta = torch.where(tiny,  torch.full_like(beta, 2.0), beta)
        return beta

    # ---- main entry ---------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        pd_scores, pd_bboxes, anc_points, stride_tensor,
        gt_labels, gt_bboxes, mask_gt,
        pd_angles: Optional[torch.Tensor] = None,
    ):
        """Assign GT boxes to anchors.

        Parameters
        ----------
        pd_scores    : (B, N, C) sigmoid probabilities
        pd_bboxes    : (B, N, 4) predicted XYXY boxes
        anc_points   : (N, 2)   anchor centres
        stride_tensor: (N, 1)   per-anchor strides
        gt_labels    : (B, G, 1)
        gt_bboxes    : (B, G, 4) GT XYXY boxes
        mask_gt      : (B, G, 1) valid GT mask
        pd_angles    : (B, N) optional predicted OBB angles (radians)
                       If provided, ProBIoU replaces CIoU in alignment.

        Returns
        -------
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx
        """
        device = pd_scores.device
        bsz, num_anchors, num_classes = pd_scores.shape
        num_gt = gt_bboxes.shape[1]

        if num_gt == 0:
            return (
                torch.full((bsz, num_anchors), num_classes, device=device, dtype=torch.long),
                torch.zeros(bsz, num_anchors, 4, device=device),
                torch.zeros(bsz, num_anchors, num_classes, device=device),
                torch.zeros(bsz, num_anchors, device=device, dtype=torch.bool),
                torch.zeros(bsz, num_anchors, device=device, dtype=torch.long),
            )

        # CUDA OOM guard: fall back to CPU if GPU memory is exhausted
        try:
            return self._inner_assign(
                pd_scores, pd_bboxes, anc_points, stride_tensor,
                gt_labels, gt_bboxes, mask_gt, pd_angles,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            # CUDA OOM → CPU fallback is ~60-200s/batch (vs ~5ms on GPU). If this
            # fires repeatedly the GPU is out of memory (shared GPU / batch too
            # large) and it is the dominant training-speed bottleneck. Make it LOUD.
            import warnings
            warnings.warn(
                '[STAL] CUDA OOM → CPU fallback (SLOW ~60-200s/batch). '
                'Free GPU memory or reduce BATCH_SIZE — this dominates epoch time.',
                RuntimeWarning, stacklevel=2)
            print('[STAL][OOM-FALLBACK] computing assignment on CPU (SLOW)', flush=True)
            torch.cuda.empty_cache()
            cpu_result = self._inner_assign(
                pd_scores.cpu(), pd_bboxes.cpu(),
                anc_points.cpu(), stride_tensor.cpu(),
                gt_labels.cpu(), gt_bboxes.cpu(), mask_gt.cpu(),
                pd_angles.cpu() if pd_angles is not None else None,
            )
            return tuple(t.to(device) if isinstance(t, torch.Tensor) else t
                         for t in cpu_result)

    def _inner_assign(
        self,
        pd_scores, pd_bboxes, anc_points, stride_tensor,
        gt_labels, gt_bboxes, mask_gt,
        pd_angles: Optional[torch.Tensor],
    ):
        mask_in_gts = self._candidates_in_gts(
            anc_points, stride_tensor.squeeze(-1), gt_bboxes, mask_gt
        )
        align_metric, overlaps = self._box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes,
            mask_in_gts * mask_gt,
            pd_angles=pd_angles,
        )
        mask_topk = self._topk_mask(align_metric, gt_bboxes, mask_gt)
        mask_pos  = mask_topk * mask_in_gts * mask_gt
        target_gt_idx, fg_mask, mask_pos = self._resolve_ambiguity(mask_pos, overlaps)
        target_labels, target_bboxes, target_scores = self._build_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(dim=-1, keepdim=True)
        pos_iou   = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm      = (align_metric * pos_iou / (pos_align + self.eps)).amax(dim=1).unsqueeze(-1)
        target_scores = (target_scores * norm).clamp(0.0, 1.0)
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # ---- candidate selection (unchanged from loss_new.py) -------------------

    def _candidates_in_gts(self, anc_points, anchor_strides, gt_bboxes, mask_gt):
        bsz, num_gt, _ = gt_bboxes.shape
        num_anchors = anc_points.shape[0]

        gt_x1 = gt_bboxes[..., 0]; gt_y1 = gt_bboxes[..., 1]
        gt_x2 = gt_bboxes[..., 2]; gt_y2 = gt_bboxes[..., 3]
        gt_w  = (gt_x2 - gt_x1).clamp(min=0)
        gt_h  = (gt_y2 - gt_y1).clamp(min=0)
        gt_cx = (gt_x1 + gt_x2) / 2.0
        gt_cy = (gt_y1 + gt_y2) / 2.0
        short_side = torch.minimum(gt_w, gt_h)

        expand, tiny, small = self._adaptive_expand(gt_w, gt_h)
        half_expand = expand / 2.0
        eff_x1 = torch.minimum(gt_x1, gt_cx - half_expand)
        eff_y1 = torch.minimum(gt_y1, gt_cy - half_expand)
        eff_x2 = torch.maximum(gt_x2, gt_cx + half_expand)
        eff_y2 = torch.maximum(gt_y2, gt_cy + half_expand)

        ax = anc_points[:, 0]; ay = anc_points[:, 1]
        stride = anchor_strides
        in_x = (ax[None, None, :] >= eff_x1[..., None]) & (ax[None, None, :] <= eff_x2[..., None])
        in_y = (ay[None, None, :] >= eff_y1[..., None]) & (ay[None, None, :] <= eff_y2[..., None])
        inside_box = in_x & in_y

        radius_factor = torch.full_like(short_side, 1.5)
        radius_factor = torch.where(small, torch.full_like(radius_factor, 2.0), radius_factor)
        radius_factor = torch.where(tiny,  torch.full_like(radius_factor, 2.5), radius_factor)
        rx = radius_factor[..., None] * stride[None, None, :]
        ry = rx
        center_prior  = (ax[None, None, :] - gt_cx[..., None]).abs() <= rx
        center_prior &= (ay[None, None, :] - gt_cy[..., None]).abs() <= ry

        level_ok = torch.ones(bsz, num_gt, num_anchors, device=gt_bboxes.device, dtype=torch.bool)
        level_ok &= mask_gt[..., 0].bool().unsqueeze(-1)
        level_ok &= torch.where(
            tiny.unsqueeze(-1),
            stride[None, None, :] <= (2.0 * self.min_stride),
            torch.ones_like(level_ok),
        )
        level_ok &= torch.where(
            small.unsqueeze(-1),
            stride[None, None, :] <= (4.0 * self.min_stride),
            torch.ones_like(level_ok),
        )

        candidate = (inside_box | center_prior) & level_ok

        # Nearest-anchor fallback for tiny/small GTs — vectorised
        dist = ((ax[None, None, :] - gt_cx[..., None]) ** 2 +
                (ay[None, None, :] - gt_cy[..., None]) ** 2).sqrt()
        norm_dist = dist / (stride[None, None, :] + self.eps)
        norm_dist = torch.where(level_ok, norm_dist, torch.full_like(norm_dist, 1e8))

        min_needed = torch.full_like(short_side, float(self.min_candidates)).long()
        min_needed = torch.where(tiny,  min_needed + 4, min_needed)
        min_needed = torch.where(small, min_needed + 2, min_needed)

        # Vectorised fallback: add nearest anchors for (b,g) pairs below min_needed.
        # The old Python loop called .item() up to 2×bsz×num_gt times per batch,
        # each forcing a GPU-CPU sync. With batch=64 and ~50 GT boxes that was
        # ~6400 syncs/batch ≈ 13 s/batch. Now: at most 2 syncs (any() + max()).
        current_count = candidate.sum(-1).long()                      # (bsz, num_gt)
        needs_fb = (current_count < min_needed) & mask_gt[..., 0].bool()
        if needs_fb.any():
            max_k = int(min_needed.max().clamp(max=num_anchors).item())
            top_idx = torch.topk(norm_dist, max_k, dim=-1, largest=False).indices
            rank  = torch.arange(max_k, device=gt_bboxes.device)[None, None, :]
            valid = needs_fb.unsqueeze(-1) & (rank < min_needed.unsqueeze(-1))
            b_i, g_i, k_i = valid.nonzero(as_tuple=True)
            if b_i.numel() > 0:
                candidate[b_i, g_i, top_idx[b_i, g_i, k_i]] = True

        return candidate.float()

    # ---- box metrics with ProBIoU support -----------------------------------

    def _box_metrics(
        self,
        pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask,
        pd_angles: Optional[torch.Tensor] = None,
    ):
        bsz, num_anchors, num_classes = pd_scores.shape
        num_gt = gt_bboxes.shape[1]
        device = pd_scores.device

        overlaps     = torch.zeros(bsz, num_gt, num_anchors, dtype=pd_bboxes.dtype, device=device)
        bbox_scores  = torch.zeros(bsz, num_gt, num_anchors, dtype=pd_scores.dtype, device=device)
        mask_bool    = mask.bool()

        cls_idx   = gt_labels.squeeze(-1).long()
        pd_t      = pd_scores.permute(0, 2, 1)
        cls_idx_e = cls_idx.unsqueeze(-1).expand(-1, -1, num_anchors)
        all_scores = torch.gather(pd_t, 1, cls_idx_e)
        bbox_scores[mask_bool] = all_scores[mask_bool]

        pd_exp = pd_bboxes.unsqueeze(1).expand(-1, num_gt, -1, -1)
        gt_exp = gt_bboxes.unsqueeze(2).expand(-1, -1, num_anchors, -1)
        pd_sel = pd_exp[mask_bool]
        gt_sel = gt_exp[mask_bool]

        if pd_sel.numel() > 0:
            if pd_angles is not None:
                # ProBIoU: convert selected XYXY boxes + angles to xywhr
                # pd_sel / gt_sel are (Nvalid, 4)
                def xyxy2xywh(b):
                    cx = (b[:, 0] + b[:, 2]) * 0.5
                    cy = (b[:, 1] + b[:, 3]) * 0.5
                    w  = (b[:, 2] - b[:, 0]).clamp(min=0)
                    h  = (b[:, 3] - b[:, 1]).clamp(min=0)
                    return cx, cy, w, h

                cx_pd, cy_pd, w_pd, h_pd = xyxy2xywh(pd_sel)
                # Gather matching predicted angles for valid (b,g,n) triples
                # mask_bool is (bsz, num_gt, num_anchors) → expand angles
                ang_exp = pd_angles.unsqueeze(1).expand(-1, num_gt, -1)  # (bsz, num_gt, N)
                ang_sel = ang_exp[mask_bool]                               # (Nvalid,)
                pd_xywhr = torch.stack([cx_pd, cy_pd, w_pd, h_pd, ang_sel], dim=-1)

                cx_gt, cy_gt, w_gt, h_gt = xyxy2xywh(gt_sel)
                # Pseudo-GT angle: atan2(h, w)  ∈ [0, π/2] ⊂ [−π/4, 3π/4]
                ang_gt_pseudo = torch.atan2(h_gt.clamp(min=1e-6), w_gt.clamp(min=1e-6))
                gt_xywhr = torch.stack([cx_gt, cy_gt, w_gt, h_gt, ang_gt_pseudo], dim=-1)

                overlaps[mask_bool] = probiou(pd_xywhr, gt_xywhr).clamp(min=0)
            else:
                overlaps[mask_bool] = bbox_iou_ciou(pd_sel, gt_sel).clamp(min=0)

        gt_w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)
        gt_h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)
        short_side = torch.minimum(gt_w, gt_h)
        beta_map   = self._adaptive_beta(short_side).unsqueeze(-1)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.clamp(min=self.eps).pow(beta_map)
        return align_metric, overlaps

    # ---- vectorised topk mask -----------------------------------------------

    def _topk_mask(self, align_metric, gt_bboxes, mask_gt):
        """Vectorised top-k anchor mask — no Python for-loop over (b, g) pairs.

        Uses torch.topk for all (b,g) pairs at once, then masks out entries
        whose rank exceeds the per-(b,g) adaptive topk value.  A nonzero()
        scatter fills the output without Python iteration.
        """
        bsz, num_gt, num_anchors = align_metric.shape
        device = align_metric.device

        gt_w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)
        gt_h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)
        short_side = torch.minimum(gt_w, gt_h)
        topk_map, _, _ = self._adaptive_topk(short_side)   # (bsz, num_gt) long

        max_k = min(int(topk_map.max().item()), num_anchors)
        if max_k <= 0:
            return torch.zeros(bsz, num_gt, num_anchors, device=device)

        # Get top max_k values and their anchor indices for every (b, g)
        top_vals, top_idx = torch.topk(align_metric, max_k, dim=-1, largest=True)
        # top_vals, top_idx : (bsz, num_gt, max_k)

        # Rank tensor: entry k is valid only if k < topk_map[b,g]
        rank      = torch.arange(max_k, device=device)[None, None, :]   # (1, 1, max_k)
        rank_mask = rank < topk_map.unsqueeze(-1)                        # (bsz, num_gt, max_k)
        val_mask  = top_vals > 0
        gt_mask   = mask_gt[..., 0].bool().unsqueeze(-1)                 # (bsz, num_gt, 1)
        valid     = rank_mask & val_mask & gt_mask                       # (bsz, num_gt, max_k)

        # Scatter valid entries into the output tensor
        out = torch.zeros(bsz, num_gt, num_anchors, device=device, dtype=torch.float32)
        b_i, g_i, k_i = valid.nonzero(as_tuple=True)
        if b_i.numel() > 0:
            out[b_i, g_i, top_idx[b_i, g_i, k_i]] = 1.0
        return out

    # ---- ambiguity resolution and target building (unchanged) ---------------

    def _resolve_ambiguity(self, mask_pos, overlaps):
        fg_mask = mask_pos.sum(dim=1)
        if fg_mask.max() > 1:
            multi   = (fg_mask.unsqueeze(1) > 1).expand_as(mask_pos)
            best_gt = overlaps.argmax(dim=1, keepdim=True)
            is_best = torch.zeros_like(mask_pos).scatter_(1, best_gt, 1.0)
            mask_pos = torch.where(multi, is_best, mask_pos)
            fg_mask  = mask_pos.sum(dim=1)
        target_gt_idx = mask_pos.argmax(dim=1)
        return target_gt_idx, fg_mask, mask_pos

    def _build_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        bsz, num_anchors = target_gt_idx.shape
        num_classes = self.num_classes
        num_gt      = gt_bboxes.shape[1]
        device      = gt_labels.device

        batch_idx = torch.arange(bsz, device=device).unsqueeze(-1)
        flat_idx  = target_gt_idx + batch_idx * num_gt

        # reshape (not view): gt_* may be a non-contiguous slice after GT clipping
        target_labels = gt_labels.long().reshape(-1)[flat_idx.reshape(-1)].reshape(bsz, num_anchors)
        target_labels = target_labels.clamp(0, num_classes - 1)
        target_bboxes = gt_bboxes.reshape(-1, 4)[flat_idx.reshape(-1)].reshape(bsz, num_anchors, 4)

        target_scores = torch.zeros(bsz, num_anchors, num_classes, dtype=torch.float32, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1.0)
        fg_exp = fg_mask.unsqueeze(-1).expand_as(target_scores)
        target_scores = torch.where(fg_exp > 0, target_scores, torch.zeros_like(target_scores))
        return target_labels, target_bboxes, target_scores


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def _focal_bce_soft(logits, targets, alpha=0.25, gamma=2.0, class_weights=None):
    # Clamp targets to valid BCE range
    targets = targets.clamp(0.0, 1.0)
    prob  = torch.sigmoid(logits)
    bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t   = prob * targets + (1.0 - prob) * (1.0 - targets)
    a_t   = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal = a_t * ((1.0 - p_t).clamp(min=1e-6) ** gamma)
    loss  = focal * bce
    # Apply class weights to both positives and negatives
    if class_weights is not None:
        cw   = class_weights.view(1, 1, -1).to(loss.device, loss.dtype)
        loss = loss * cw
    return loss.sum()


def _centerness_targets(anc_points, target_bboxes, fg_mask):
    bsz, num_anchors, _ = target_bboxes.shape
    cx = anc_points[:, 0].unsqueeze(0).expand(bsz, -1)
    cy = anc_points[:, 1].unsqueeze(0).expand(bsz, -1)
    l_ = (cx - target_bboxes[..., 0])[fg_mask].clamp(min=1e-6)
    t_ = (cy - target_bboxes[..., 1])[fg_mask].clamp(min=1e-6)
    r_ = (target_bboxes[..., 2] - cx)[fg_mask].clamp(min=1e-6)
    b_ = (target_bboxes[..., 3] - cy)[fg_mask].clamp(min=1e-6)
    ctr = torch.sqrt((torch.min(l_, r_) / torch.max(l_, r_)) *
                     (torch.min(t_, b_) / torch.max(t_, b_)) + 1e-8)
    return ctr.clamp(0.0, 1.0)


def _prog_weight(epoch: int, prog_loss_epochs: int) -> float:
    if prog_loss_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / prog_loss_epochs)


def _angle_loss_sin2(
    ang_pred: torch.Tensor,
    ang_gt: torch.Tensor,
    gt_w: torch.Tensor,
    gt_h: torch.Tensor,
    num_fg: float,
) -> torch.Tensor:
    """Angle loss: sin²(2·Δθ) weighted by aspect-ratio relevance.

    Why sin²(2·Δθ)?
    - π/2-periodic: correctly handles 0 ↔ π/2 wrap-around.
    - Differentiable everywhere; gradient ∝ sin(4Δθ) which pushes toward
      the nearest in-range solution.
    - Previous smooth_L1 on angles can produce large gradients near ±π/4
      where small prediction errors wrap to large loss values.

    Aspect-ratio weighting: downweights angle loss for near-square objects
    where orientation is ambiguous and prediction error has less impact on
    box quality.
    """
    with torch.no_grad():
        ar = (gt_w / gt_h.clamp(min=1e-6)).clamp(min=1e-6)
        ar = torch.max(ar, 1.0 / ar.clamp(min=1e-6))   # ensure ≥ 1
        # ar=1 (square) → weight≈0;  ar=4 (elongated) → weight≈0.78
        ar_weight = 1.0 - torch.exp(-((ar - 1.0) / 2.0))

    return (torch.sin(2.0 * (ang_pred - ang_gt)).pow(2) * ar_weight).sum() / num_fg


# ---------------------------------------------------------------------------
# Main loss function
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── VariFocal Loss ─────────────────────────────────────────────────────────────

def _varifocal_loss(logits, targets, alpha=0.75, gamma=2.0, class_weights=None,
                    cw_apply_neg=False, q_floor=0.0):
    """VariFocal Loss (VFL).

    Parameters
    ----------
    logits       : (B, N, C) raw classification logits
    targets      : (B, N, C) soft quality scores from STAL in [0, 1]
    alpha        : weight for negative focal term  (0.75 for VFL, vs 0.25 for focal)
    gamma        : focusing exponent for negatives
    class_weights: (C,) per-class weights, or None
    cw_apply_neg : if True, class weights also scale the negative term. Default
                   False — see note below.

    Returns
    -------
    Scalar loss (unreduced sum — caller divides by num_fg).
    """
    # Clamp targets to valid BCE range — STAL normalization can produce >1.0
    targets = targets.clamp(0.0, 1.0)

    prob = torch.sigmoid(logits)
    pos  = targets > 0

    # Positive: weight = quality score q  →  trains head to predict IoU-aware score
    # Negative: weight = alpha * p^gamma  →  standard focal suppression
    pos_weight = targets.clamp(min=q_floor) if q_floor > 0 else targets
    focal_weight = torch.where(
        pos,
        pos_weight,                     # q (quality score) for positives, floored
        alpha * prob.pow(gamma),        # alpha * p^gamma   for negatives
    )

    bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    loss = bce * focal_weight

    # Class weights. By default applied to POSITIVES ONLY: this up-weights rare
    # positives (recall) without inflating the rare-class negative term across
    # the vastly more numerous background anchors (which would suppress recall —
    # the opposite of the deployment goal). Set cw_apply_neg=True to weight both
    # (favours precision / FP suppression).
    if class_weights is not None:
        cw = class_weights.view(1, 1, -1).to(loss.device, loss.dtype)
        if cw_apply_neg:
            loss = loss * cw
        else:
            loss = torch.where(pos, loss * cw, loss)

    return loss.sum()


# ── Distribution Focal Loss (DFL) ─────────────────────────────────────────────

def _dfl_loss(pred_dist, target_ltrb, stride_fg, reg_max=16):
    """Distribution Focal Loss for box regression.

    Parameters
    ----------
    pred_dist   : (Nfg, 4*(reg_max+1)) raw logits from DFL head
    target_ltrb : (Nfg, 4) target LTRB in pixel units
    stride_fg   : (Nfg,) per-anchor stride for normalization to [0, reg_max]
    reg_max     : number of discrete bins minus 1

    Returns
    -------
    Scalar loss (unreduced sum — caller divides by num_fg).
    """
    # Normalize targets to [0, reg_max] range (stride units)
    target_norm = target_ltrb / stride_fg.unsqueeze(-1)
    target_norm = target_norm.clamp(0, reg_max - 0.01)

    # Discrete lower/upper bounds for soft two-bin cross-entropy
    tl = target_norm.long()             # floor
    tr = tl + 1                         # ceil
    wl = tr.float() - target_norm       # weight for lower bin
    wr = 1.0 - wl                       # weight for upper bin

    # Reshape pred to (Nfg, 4, reg_max+1) and compute log-softmax per side
    pred = pred_dist.reshape(-1, 4, reg_max + 1)
    log_prob = F.log_softmax(pred, dim=2)

    # Cross-entropy with soft two-bin target
    loss_l = log_prob.gather(2, tl.unsqueeze(2)).squeeze(2)
    loss_r = log_prob.gather(2, tr.clamp(max=reg_max).unsqueeze(2)).squeeze(2)
    loss = -(wl * loss_l + wr * loss_r)
    return loss.sum()


# ── Auxiliary Decoder Loss (Hungarian matching) ────────────────────────────────

def _aux_decoder_loss(aux_out, batch_boxes, batch_labels, image_size, num_classes):
    """Hungarian-matched auxiliary decoder loss (training only).

    Parameters
    ----------
    aux_out      : dict with 'aux_cls' (B, Q, C) and 'aux_boxes' (B, Q, 4) normalized cxcywh
    batch_boxes  : list of (N_i, 4) boxes in relative xywh format
    batch_labels : list of (N_i,) integer labels
    image_size   : (H, W)
    num_classes  : number of classes

    Returns
    -------
    Scalar loss (already normalized).
    """
    if not _HAS_SCIPY:
        return torch.tensor(0.0, device=aux_out['aux_cls'].device)

    aux_cls = aux_out['aux_cls']        # (B, Q, C) raw logits
    aux_boxes = aux_out['aux_boxes']    # (B, Q, 4) normalized cxcywh (sigmoid)
    B, Q, C = aux_cls.shape
    img_h, img_w = image_size
    device = aux_cls.device

    total_loss = torch.zeros(1, device=device, requires_grad=True)
    num_matched = 0

    for b in range(B):
        boxes_rel = batch_boxes[b]
        if boxes_rel.ndim != 2:
            boxes_rel = boxes_rel.reshape(-1, 4)
        labs = batch_labels[b]
        n_gt = boxes_rel.shape[0]
        if n_gt == 0:
            continue

        # Convert GT from relative xywh to normalized cxcywh
        gt_cx = boxes_rel[:, 0] + boxes_rel[:, 2] * 0.5
        gt_cy = boxes_rel[:, 1] + boxes_rel[:, 3] * 0.5
        gt_w  = boxes_rel[:, 2]
        gt_h  = boxes_rel[:, 3]
        gt_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=-1)  # (N, 4)

        # Cost matrix: L1 box cost + classification cost
        pred_prob = aux_cls[b].sigmoid()                       # (Q, C)
        pred_box  = aux_boxes[b]                               # (Q, 4)

        # Box L1 cost: (Q, N)
        box_cost = torch.cdist(pred_box, gt_cxcywh.to(device), p=1)  # (Q, N)

        # Cls cost: negative log-prob of the GT class
        cls_cost = -pred_prob[:, labs.long()]                  # (Q, N)

        cost = 5.0 * box_cost + 1.0 * cls_cost                # (Q, N)
        cost_np = cost.detach().cpu().numpy()

        # Hungarian matching
        row_idx, col_idx = linear_sum_assignment(cost_np)
        row_idx = torch.tensor(row_idx, device=device, dtype=torch.long)
        col_idx = torch.tensor(col_idx, device=device, dtype=torch.long)

        # Classification loss (focal-style) on matched queries
        matched_cls = aux_cls[b, row_idx]                      # (M, C)
        matched_labs = labs[col_idx].long()                     # (M,)
        cls_target = torch.zeros_like(matched_cls)
        cls_target[torch.arange(len(col_idx)), matched_labs] = 1.0
        cls_loss = F.binary_cross_entropy_with_logits(
            matched_cls, cls_target, reduction='sum')

        # Box L1 loss on matched queries
        matched_pred_box = pred_box[row_idx]                   # (M, 4)
        matched_gt_box = gt_cxcywh[col_idx].to(device)        # (M, 4)
        reg_loss = F.l1_loss(matched_pred_box, matched_gt_box, reduction='sum')

        total_loss = total_loss + cls_loss + 5.0 * reg_loss
        num_matched += len(row_idx)

    return total_loss / max(num_matched, 1)


# ── Main loss ──────────────────────────────────────────────────────────────────

def compute_loss(
    outputs,
    batch_boxes,
    batch_labels,
    image_size,
    strides,
    focal_alpha: float = 0.25,    # kept for API compatibility; not used by VFL
    focal_gamma: float = 2.0,
    weight_reg: float = 1.0,
    weight_ctr: float = 1.0,
    weight_angle: float = 0.2,
    tal_topk: int = 10,
    tal_alpha: float = 0.5,
    tal_beta: float = 4.0,
    epoch: int = 0,
    num_epochs: int = 100,
    prog_loss_epochs: int = 10,
    class_weights=None,
    vfl_alpha: float = 0.75,
    vfl_cw_neg: bool = False,
    vfl_q_floor: float = 0.0,
    # Phase C params
    reg_max: int = 0,
    use_centerness: bool = True,
    aux_loss_weight: float = 0.0,
    **_,
):
    """Compute detection loss with VariFocal cls + optional DFL reg + aux decoder.

    Returns
    -------
    (total, cls_loss, reg_loss, ctr_loss, angle_loss)  — all scalar tensors
    """
    device      = outputs['cls'][0].device
    bsz         = outputs['cls'][0].shape[0]
    num_classes = outputs['cls'][0].shape[1]
    img_h, img_w = image_size

    cls_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(bsz, -1, num_classes)
                          for x in outputs['cls']], dim=1)
    reg_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(bsz, -1, 4)
                          for x in outputs['reg']], dim=1)
    ctr_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(bsz, -1, 1)
                          for x in outputs['ctr']], dim=1)

    has_angle = 'angle' in outputs
    if has_angle:
        ang_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(bsz, -1, 1)
                               for x in outputs['angle']], dim=1)

    # DFL raw logits (only present when reg_max > 0)
    has_dfl = 'dfl_raw' in outputs
    if has_dfl:
        dfl_ch = outputs['dfl_raw'][0].shape[1]  # 4*(reg_max+1)
        dfl_flat = torch.cat([x.permute(0, 2, 3, 1).reshape(bsz, -1, dfl_ch)
                              for x in outputs['dfl_raw']], dim=1)

    anc_points, stride_tensor = make_anchors(outputs['cls'], strides)

    l_, t_, r_, b_ = reg_flat.unbind(-1)
    cx = anc_points[:, 0]
    cy = anc_points[:, 1]
    pd_bboxes    = torch.stack([cx - l_, cy - t_, cx + r_, cy + b_], dim=-1)
    pd_scores    = cls_flat.sigmoid()
    pd_angles_2d = ang_flat.squeeze(-1) if has_angle else None

    def _to_2d(b):
        return b.reshape(-1, 4) if b.ndim != 2 else b

    max_gt = max((_to_2d(b).shape[0] for b in batch_boxes), default=0)
    if max_gt == 0:
        z = torch.zeros(1, device=device, requires_grad=True)
        return z, z.detach(), z.detach(), z.detach(), z.detach()

    gt_bboxes_list, gt_labels_list, mask_gt_list = [], [], []
    for i in range(bsz):
        boxes_rel = _to_2d(batch_boxes[i])
        labs      = batch_labels[i]
        ni        = boxes_rel.shape[0]

        x1 = boxes_rel[:, 0] * img_w;  y1 = boxes_rel[:, 1] * img_h
        x2 = x1 + boxes_rel[:, 2] * img_w
        y2 = y1 + boxes_rel[:, 3] * img_h
        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

        pad = max_gt - ni
        if pad > 0:
            boxes_abs = torch.cat([boxes_abs, torch.zeros(pad, 4, device=device)], dim=0)
            labs = torch.cat([labs, torch.zeros(pad, device=device, dtype=labs.dtype)], dim=0)

        mask = torch.zeros(max_gt, 1, device=device)
        mask[:ni] = 1.0
        gt_bboxes_list.append(boxes_abs)
        gt_labels_list.append(labs.unsqueeze(-1))
        mask_gt_list.append(mask)

    gt_bboxes = torch.stack(gt_bboxes_list, dim=0)
    gt_labels = torch.stack(gt_labels_list, dim=0)
    mask_gt   = torch.stack(mask_gt_list,   dim=0)

    # Clip max GT boxes per batch to prevent CUDA OOM on dense mosaic batches.
    MAX_GT_CLIP = 100
    if gt_bboxes.shape[1] > MAX_GT_CLIP:
        gt_bboxes = gt_bboxes[:, :MAX_GT_CLIP].contiguous()
        gt_labels = gt_labels[:, :MAX_GT_CLIP].contiguous()
        mask_gt   = mask_gt[:,   :MAX_GT_CLIP].contiguous()

    min_stride  = float(strides[0])
    stal_stride = float(strides[1]) if len(strides) > 1 else 2.0 * min_stride
    assigner    = STALTaskAlignedAssignerOBB(
        topk=tal_topk, num_classes=num_classes,
        alpha=tal_alpha, beta=tal_beta,
        min_stride=min_stride, stal_stride=stal_stride,
    )
    target_labels, target_bboxes, target_scores, fg_mask, _ = assigner(
        pd_scores, pd_bboxes, anc_points, stride_tensor,
        gt_labels, gt_bboxes, mask_gt,
        pd_angles=pd_angles_2d,
    )

    num_fg = fg_mask.float().sum().clamp(min=1.0)

    # ── VFL classification loss ───────────────────────────────────────────────
    cls_loss = _varifocal_loss(
        cls_flat, target_scores,
        alpha=vfl_alpha, gamma=focal_gamma,
        class_weights=class_weights,
        cw_apply_neg=vfl_cw_neg,
        q_floor=vfl_q_floor,
    ) / num_fg

    # ── Regression + centerness + angle losses ────────────────────────────────
    if fg_mask.any():
        pd_bboxes_pos  = pd_bboxes[fg_mask]
        tgt_bboxes_pos = target_bboxes[fg_mask]

        ciou_term = ciou_loss(pd_bboxes_pos, tgt_bboxes_pos) / num_fg

        stride_fg = stride_tensor.squeeze(-1).unsqueeze(0).expand(bsz, -1)[fg_mask]

        if has_dfl and reg_max > 0:
            # DFL loss replaces L1 term
            dfl_pos = dfl_flat[fg_mask]
            # Compute target LTRB from anchor points and assigned GT boxes
            anc_x = anc_points[:, 0].unsqueeze(0).expand(bsz, -1)[fg_mask]
            anc_y = anc_points[:, 1].unsqueeze(0).expand(bsz, -1)[fg_mask]
            tgt_l = (anc_x - tgt_bboxes_pos[:, 0]).clamp(min=0)
            tgt_t = (anc_y - tgt_bboxes_pos[:, 1]).clamp(min=0)
            tgt_r = (tgt_bboxes_pos[:, 2] - anc_x).clamp(min=0)
            tgt_b = (tgt_bboxes_pos[:, 3] - anc_y).clamp(min=0)
            tgt_ltrb = torch.stack([tgt_l, tgt_t, tgt_r, tgt_b], dim=-1)
            dfl_term = _dfl_loss(dfl_pos, tgt_ltrb, stride_fg,
                                 reg_max=reg_max) / num_fg
            reg_loss = 0.7 * ciou_term + 0.3 * dfl_term
        else:
            # Fallback: stride-normalized L1
            stride_norm = stride_fg.unsqueeze(-1).expand_as(pd_bboxes_pos)
            l1_term = F.l1_loss(pd_bboxes_pos / stride_norm,
                                tgt_bboxes_pos / stride_norm,
                                reduction='sum') / num_fg
            reg_loss = 0.7 * ciou_term + 0.3 * l1_term

        # Centerness loss (optional — skip when VFL already provides IoU-awareness)
        if use_centerness:
            ctr_pred = ctr_flat[fg_mask].squeeze(-1)
            ctr_tgt  = _centerness_targets(anc_points, target_bboxes, fg_mask)
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred, ctr_tgt, reduction='sum') / num_fg
        else:
            ctr_loss = torch.tensor(0.0, device=device)

        if has_angle:
            ang_pred  = ang_flat[fg_mask].squeeze(-1)
            gt_w_pos  = (tgt_bboxes_pos[:, 2] - tgt_bboxes_pos[:, 0]).clamp(min=1.0)
            gt_h_pos  = (tgt_bboxes_pos[:, 3] - tgt_bboxes_pos[:, 1]).clamp(min=1.0)
            angle_gt  = torch.atan2(gt_h_pos, gt_w_pos)
            angle_loss = _angle_loss_sin2(ang_pred, angle_gt, gt_w_pos, gt_h_pos, num_fg)
        else:
            angle_loss = torch.tensor(0.0, device=device)
    else:
        reg_loss   = torch.tensor(0.0, device=device)
        ctr_loss   = torch.tensor(0.0, device=device)
        angle_loss = torch.tensor(0.0, device=device)

    prog_w = _prog_weight(epoch, prog_loss_epochs)
    total  = cls_loss + (weight_reg * reg_loss + weight_ctr * ctr_loss +
                         weight_angle * angle_loss) * prog_w

    # ── Auxiliary decoder loss (training only) ────────────────────────────────
    if 'aux' in outputs and aux_loss_weight > 0:
        aux_loss = _aux_decoder_loss(
            outputs['aux'], batch_boxes, batch_labels,
            image_size, num_classes)
        total = total + aux_loss_weight * aux_loss

    return total, cls_loss, reg_loss, ctr_loss, angle_loss
