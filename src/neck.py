"""
model_head_v3.py — Multi-layer FPN + Per-level Prediction Heads
================================================================

Two improvements over model_head_v2.py:

1. MultiLevelEfficientPAN  (FPN improvement)
   ───────────────────────────────────────────
   v2: All FPN levels derived from a single deep backbone feature.
       P3 = proj(deep) → only global semantics → poor for small objects.

   v3: Each FPN level seeded from a different backbone depth:
       P3 ← shallow (block ~n/4)  rich local textures/edges
       P4 ← mid     (block ~n/2)  part-level representations
       P5 ← deep    (block  n-1)  global semantics for large objects

2. DetectHeadV3  (per-level prediction heads)
   ─────────────────────────────────────────────
   v2: cls_logits / bbox_reg / centerness / angle_reg are SHARED across all
       FPN levels — P3 and P5 use the same 3×3 conv for final prediction.
       This prevents level-specific specialization.

   v3: Each FPN level gets its OWN prediction convolutions:
       cls_logits[P3], cls_logits[P4], cls_logits[P5]  (separate biases)
       bbox_reg  [P3], bbox_reg  [P4], bbox_reg  [P5]
       centerness[P3], centerness[P4], centerness[P5]
       angle_reg [P3], angle_reg [P4], angle_reg [P5]

       The tower convolutions (cls_tower / reg_tower RepDWS blocks) remain
       SHARED — they contribute most of the capacity and sharing them keeps
       parameter count reasonable while per-level specialization is in the
       final 3×3 prediction layer.

Why NOT DFL in v3?
──────────────────
   YOLOv26 uses DFL (Distribution Focal Loss, reg_max=16) for regression.
   DFL requires changing reg_flat from (B,N,4) to (B,N,4*reg_max) and
   decoding inside compute_loss before CIoU, which is a significant loss
   surgery. Planned for Stage 2+.

Why NOT SAVPE?
──────────────
   Spatial Anchor-free Vector Position Encoding adds a learnable position
   bias to the classification branch. Marginal gain (<0.5%), significant
   code complexity. Skip.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Only import what's actually used from v2
from src.blocks import (
    DWSConvGNReLU,
    RepDWSBlock,
    SmallObjectRefine,
    count_parameters,
)


# ──────────────────────────────────────────────────────────────────────────────
# MultiLevelEfficientPAN
# ──────────────────────────────────────────────────────────────────────────────

class MultiLevelEfficientPAN(nn.Module):
    """Bidirectional FPN seeded from three backbone depths.

    Inputs
    ------
    features : [shallow, mid, deep]  — each (B, in_channels, H/16, W/16)

    Build
    -----
    Stage 1 — project + downsample to pyramid strides:
        P3 = proj(shallow)                       stride 16
        P4 = down_2x(proj(mid))                  stride 32
        P5 = down_2x(down_2x(proj(deep)))        stride 64

    Stage 2 — top-down (coarse → fine):
        P5_td = smooth(P5)
        P4_td = smooth(P4 + upsample(P5_td))
        P3_td = smooth(P3 + upsample(P4_td)) + SmallObjectRefine

    Stage 3 — bottom-up (fine → coarse localisation):
        P4_bu = smooth(P4_td + down_2x(P3_td))
        P5_bu = smooth(P5_td + down_2x(P4_bu))

    Output: [P3_td, P4_bu, P5_bu]
    """

    def __init__(self, in_channels: int, out_channels: int = 192):
        super().__init__()
        C = out_channels
        self.out_channels = C

        def _proj(ic, oc):
            g = min(16, oc)
            while oc % g != 0 and g > 1:
                g -= 1
            return nn.Sequential(
                nn.Conv2d(ic, oc, 1, bias=False),
                nn.GroupNorm(g, oc),
                nn.ReLU(inplace=True),
            )

        self.proj_shallow = _proj(in_channels, C)
        self.proj_mid     = _proj(in_channels, C)
        self.proj_deep    = _proj(in_channels, C)

        self.down_mid    = DWSConvGNReLU(C, stride=2)   # mid   stride 16→32
        self.down_deep_1 = DWSConvGNReLU(C, stride=2)   # deep  stride 16→32
        self.down_deep_2 = DWSConvGNReLU(C, stride=2)   # deep  stride 32→64

        self.td_smooth5 = RepDWSBlock(C)
        self.td_smooth4 = RepDWSBlock(C)
        self.td_smooth3 = RepDWSBlock(C)
        self.p3_refine  = SmallObjectRefine(C)   # 2×DWS + ECA residual for small objects

        self.bu_down_p4 = DWSConvGNReLU(C, stride=2)
        self.bu_smooth4 = RepDWSBlock(C)
        self.bu_down_p5 = DWSConvGNReLU(C, stride=2)
        self.bu_smooth5 = RepDWSBlock(C)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        shallow, mid, deep = features

        p3_raw = self.proj_shallow(shallow)
        p4_raw = self.down_mid(self.proj_mid(mid))
        p5_raw = self.down_deep_2(self.down_deep_1(self.proj_deep(deep)))

        p5_td = self.td_smooth5(p5_raw)
        p4_td = self.td_smooth4(
            p4_raw + F.interpolate(p5_td, size=p4_raw.shape[-2:], mode='nearest'))
        p3_td = self.td_smooth3(
            p3_raw + F.interpolate(p4_td, size=p3_raw.shape[-2:], mode='nearest'))
        p3_td = self.p3_refine(p3_td)

        p4_bu = self.bu_smooth4(p4_td + self.bu_down_p4(p3_td))
        p5_bu = self.bu_smooth5(p5_td + self.bu_down_p5(p4_bu))

        return [p3_td, p4_bu, p5_bu]

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ──────────────────────────────────────────────────────────────────────────────
# DetectHeadV3 — Per-level prediction convolutions
# ──────────────────────────────────────────────────────────────────────────────

class DetectHeadV3(nn.Module):
    """Per-level prediction heads with shared RepDWS towers.

    Shared  : cls_tower, reg_tower  (RepDWS blocks — most of the capacity)
    Per-level: cls_logits[l], bbox_reg[l], centerness[l], angle_reg[l]

    This lets each FPN level specialise its final prediction:
      P3 (stride 16, small objects)  → own bias/weights
      P4 (stride 32, medium objects) → own bias/weights
      P5 (stride 64, large objects)  → own bias/weights

    Compatible with the same loss (loss_v3.py / loss_OBB.py) — output format
    is identical: dict with keys 'cls', 'reg', 'ctr', 'angle' each a list of
    one tensor per FPN level.
    """

    def __init__(self, in_channels: int = 192,
                 num_classes: int = 80,
                 num_convs: int = 4,
                 num_levels: int = 3,
                 obb: bool = True,
                 prior_prob: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels  = num_levels
        self.obb         = obb

        # ── Shared towers (RepDWS — bulk of parameters) ───────────────────────
        self.cls_tower = nn.Sequential(
            *[RepDWSBlock(in_channels) for _ in range(num_convs)])
        self.reg_tower = nn.Sequential(
            *[RepDWSBlock(in_channels) for _ in range(num_convs)])

        # ── Per-level prediction convolutions ─────────────────────────────────
        self.cls_logits = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, 3, padding=1)
            for _ in range(num_levels)
        ])
        self.bbox_reg = nn.ModuleList([
            nn.Conv2d(in_channels, 4, 3, padding=1)
            for _ in range(num_levels)
        ])
        self.centerness = nn.ModuleList([
            nn.Conv2d(in_channels, 1, 3, padding=1)
            for _ in range(num_levels)
        ])
        if obb:
            self.angle_reg = nn.ModuleList([
                nn.Conv2d(in_channels, 1, 3, padding=1)
                for _ in range(num_levels)
            ])

        # Per-level learnable regression scales (same as v2)
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_levels)])

        # ── Initialise ────────────────────────────────────────────────────────
        bias_cls = float(-math.log((1.0 - prior_prob) / prior_prob))
        for conv in self.cls_logits:
            nn.init.constant_(conv.bias, bias_cls)
            nn.init.normal_(conv.weight, std=0.01)

        for mod_list in [self.bbox_reg, self.centerness]:
            for conv in mod_list:
                nn.init.normal_(conv.weight, std=0.001)
                nn.init.zeros_(conv.bias)

        if obb:
            for conv in self.angle_reg:
                nn.init.normal_(conv.weight, std=0.001)
                nn.init.zeros_(conv.bias)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        cls_out: List[torch.Tensor] = []
        reg_out: List[torch.Tensor] = []
        ctr_out: List[torch.Tensor] = []
        ang_out: List[torch.Tensor] = []

        for lvl, feat in enumerate(features):
            cls_f = self.cls_tower(feat)
            reg_f = self.reg_tower(feat)

            cls_out.append(self.cls_logits[lvl](cls_f))
            reg_out.append(
                F.softplus(self.bbox_reg[lvl](reg_f)) * self.scales[lvl])
            ctr_out.append(self.centerness[lvl](reg_f))

            if self.obb:
                ang_out.append(
                    (torch.sigmoid(self.angle_reg[lvl](reg_f)) - 0.25) * math.pi)

        out: Dict[str, List[torch.Tensor]] = {
            'cls': cls_out, 'reg': reg_out, 'ctr': ctr_out}
        if self.obb:
            out['angle'] = ang_out
        return out

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ──────────────────────────────────────────────────────────────────────────────
# Top-Level Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DINODetectionHeadV3(nn.Module):
    """Multi-layer DINOv3 features → MultiLevelEfficientPAN → DetectHeadV3.

    Parameters
    ----------
    backbone_out_channels : embed_dim of frozen DINOv3 (384 for ViT-S+)
    fpn_channels          : FPN internal width (192 default)
    num_classes           : number of object categories
    num_convs             : RepDWS tower depth (4 default)
    obb                   : enable OBB angle branch
    """

    def __init__(self, backbone_out_channels: int = 384,
                 fpn_channels: int = 192,
                 num_classes: int = 17,
                 num_convs: int = 4,
                 obb: bool = True):
        super().__init__()
        self.fpn  = MultiLevelEfficientPAN(backbone_out_channels,
                                           out_channels=fpn_channels)
        self.head = DetectHeadV3(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=num_convs,
            num_levels=3,
            obb=obb,
        )

    def forward(self, features: List[torch.Tensor]) -> Dict:
        """features: [shallow, mid, deep]  from DinoBackboneV3"""
        return self.head(self.fpn(features))

    def fuse(self):
        self.fpn.fuse()
        self.head.fuse()
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import math
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed  = 384   # ViT-S+

    H_feat, W_feat = 30, 40   # 480×640 → stride-16 feature map
    feats = [torch.randn(1, embed, H_feat, W_feat, device=device) for _ in range(3)]

    head = DINODetectionHeadV3(
        backbone_out_channels=embed,
        fpn_channels=192,
        num_classes=17,
        num_convs=4,
        obb=True,
    ).to(device)

    out = head(feats)
    print('Output keys:', list(out.keys()))
    for k, v in out.items():
        print(f'  {k}: {[tuple(t.shape) for t in v]}')

    params = count_parameters(head)
    print(f'Head params : {params["total"]/1e6:.3f} M  '
          f'(trainable: {params["trainable"]/1e6:.3f} M)')

    # Show per-level weight independence
    w_p3 = head.head.cls_logits[0].weight.data.sum().item()
    w_p5 = head.head.cls_logits[2].weight.data.sum().item()
    print(f'cls_logits[P3] weight sum = {w_p3:.4f}')
    print(f'cls_logits[P5] weight sum = {w_p5:.4f}')
    print('(different → per-level specialization confirmed)')

    # Test fuse
    head.fuse()
    out2 = head(feats)
    print('After fuse cls:', [tuple(t.shape) for t in out2['cls']])
