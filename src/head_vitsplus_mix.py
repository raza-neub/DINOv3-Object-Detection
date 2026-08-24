"""
model_head_mix.py — Dual-classifier detection head for COCO + Neubie mixed training
=====================================================================================

Architecture:
    Input [shallow, mid, deep]
        ↓
    MultiLevelEfficientPAN  (shared, from model_head_v3)
        ↓
    ┌── cls path (decoupled, wider+deeper) ──────────────────────────────┐
    │   cls_entry (1×1 → cls_channels) → cls_tower (cls_convs RepDWS) → ECA │
    │       ├── cls_coco   [L × 80]  cosine classifier  (COCO auxiliary)    │
    │       └── cls_neubie [L × 16]  cosine classifier  (Neubie, deployed)  │
    └──────────────────────────────────────────────────────────────────────┘
    ┌── reg path (lightweight) ──────────────────────────────────────────┐
    │   reg_tower (num_convs RepDWS) → bbox_reg / centerness / angle_reg   │
    └──────────────────────────────────────────────────────────────────────┘

Rare-class / accuracy enhancements over the plain v3 head
---------------------------------------------------------
1. Decoupled cls/reg towers. Classification gets a **deeper and wider** tower
   (default 6 RepDWS blocks @ 256 ch) than regression (4 @ 192). Classification
   benefits from more capacity; regression is comparatively easy, so it stays
   light. A 1×1 `cls_entry` expands the 192-ch FPN features to the cls width.

2. Cosine (normalized) classifier. The final cls layer is a 1×1 cosine head:
   `scale · ⟨f̂, ŵ⟩ + bias`. L2-normalizing both the per-location feature and
   the per-class weight decouples classification from magnitude — the standard
   long-tailed-recognition fix, since rare-class weight vectors are no longer
   shrunk by the dominant classes' gradients. A learnable temperature restores
   the logit range needed by BCE/VariFocal; the per-class bias carries the
   focal-style prior. This is the main rare-class lever and it is *stabilising*
   (bounded logits, prior-correct init).

3. ECA channel attention at the cls-tower output — near-zero params, sharpens
   channel discrimination for small / rare objects.

These keep the output dict identical to v3 ({'cls','reg','ctr'[,'angle']}, one
tensor per level), so loss_v3 / decode_outputs_OBB need no changes.

The frozen DINOv3 backbone dominates latency, so the extra head capacity costs
little end-to-end FPS.

Usage:
    outputs = model_head(feats, dataset='coco')    # COCO batch  → cls_coco
    outputs = model_head(feats, dataset='neubie')  # Neubie/infer → cls_neubie
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neck import (
    MultiLevelEfficientPAN,
    DWSConvGNReLU,
    RepDWSBlock,
    count_parameters,
)
from src.blocks import (
    ConvGNReLU,
    ECABlock,
)

try:
    from torchvision.ops import DeformConv2d
    _HAS_DCN = True
except ImportError:
    _HAS_DCN = False


# ──────────────────────────────────────────────────────────────────────────────
# DCNv2 block — deformable conv with learned offsets + modulation mask
# ──────────────────────────────────────────────────────────────────────────────

class DCNv2Block(nn.Module):
    """Deformable convolution v2 with offset+mask, GroupNorm, ReLU, and residual."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        if not _HAS_DCN:
            raise ImportError('torchvision.ops.DeformConv2d required for DCNv2Block')
        k2 = kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(
            channels, 3 * k2, kernel_size, padding=kernel_size // 2)
        self.dcn = DeformConv2d(channels, channels, kernel_size,
                                padding=kernel_size // 2)
        g = min(16, channels)
        while channels % g != 0:
            g -= 1
        self.gn = nn.GroupNorm(g, channels)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)

    def forward(self, x):
        om = self.offset_mask(x)
        k2 = self.dcn.weight.shape[2] * self.dcn.weight.shape[3]
        offset = om[:, :2 * k2]
        mask = torch.sigmoid(om[:, 2 * k2:])
        return self.act(self.gn(self.dcn(x, offset, mask))) + x


# ──────────────────────────────────────────────────────────────────────────────
# Cosine (normalized) classifier — long-tail-friendly final cls layer
# ──────────────────────────────────────────────────────────────────────────────

class CosineConv2d(nn.Module):
    """1×1 cosine-similarity classifier: ``scale · ⟨f̂, ŵ⟩ + bias``.

    Normalizing both the per-location feature and the per-class weight decouples
    the decision from feature/weight magnitude — the canonical fix for
    long-tailed / rare-class recognition (rare-class weight norms stop being
    suppressed by the frequent classes). The learnable ``scale`` restores logit
    range for BCE/VFL; the per-class ``bias`` carries the focal-style prior so
    the initial positive probability is ``prior_prob`` as usual.

    Output shape matches ``nn.Conv2d(in_channels, num_classes, 1)`` → drop-in.
    """

    def __init__(self, in_channels: int, num_classes: int,
                 scale_init: float = 20.0, prior_prob: float = 0.01,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Weight: (num_classes, in_channels, 1, 1), small random init.
        self.weight = nn.Parameter(torch.randn(num_classes, in_channels, 1, 1) * 0.01)
        bias_val = float(-math.log((1.0 - prior_prob) / prior_prob))
        self.bias = nn.Parameter(torch.full((num_classes,), bias_val))
        # Learnable temperature stored in log-space → always positive.
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=1, eps=self.eps)   # unit class directions
        x = F.normalize(x, dim=1, eps=self.eps)             # unit feature per location
        cos = F.conv2d(x, w)                                # ⟨f̂, ŵ⟩ ∈ [-1, 1]
        return cos * self.log_scale.exp() + self.bias.view(1, -1, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────
# DetectHeadMix — dual classifiers, shared regression
# ──────────────────────────────────────────────────────────────────────────────

class DetectHeadMix(nn.Module):
    """Per-level prediction heads with dual classification branches.

    Shared across datasets:
        cls_entry, cls_tower, cls_eca   (cls feature extractor — wider/deeper)
        reg_tower                        (reg feature extractor — lightweight)
        bbox_reg, centerness, angle_reg  (per FPN level)

    Dataset-specific:
        cls_coco   [num_levels]  cosine (or plain) classifier — COCO auxiliary
        cls_neubie [num_levels]  cosine (or plain) classifier — Neubie final

    forward(features, dataset='neubie') selects cls_coco or cls_neubie.
    """

    def __init__(self, in_channels: int = 192,
                 num_coco_classes: int = 80,
                 num_neubie_classes: int = 16,
                 num_convs: int = 4,
                 cls_convs: int = 6,
                 cls_channels: int = 256,
                 num_levels: int = 3,
                 obb: bool = True,
                 prior_prob: float = 0.01,
                 use_eca: bool = True,
                 cosine_cls: bool = True,
                 cosine_scale: float = 20.0,
                 reg_max: int = 0,
                 use_centerness: bool = True,
                 use_dcn: bool = False):
        super().__init__()
        self.num_levels         = num_levels
        self.num_coco_classes   = num_coco_classes
        self.num_neubie_classes = num_neubie_classes
        self.obb                = obb
        self.cosine_cls         = cosine_cls
        self.cls_channels       = cls_channels
        self.reg_max            = reg_max
        self.use_centerness     = use_centerness

        # ── Classification path (decoupled, wider + deeper than reg) ──────────
        # 1×1 expansion from FPN width → cls width (Identity if equal).
        self.cls_entry = (ConvGNReLU(in_channels, cls_channels,
                                     kernel_size=1, stride=1, padding=0)
                          if cls_channels != in_channels else nn.Identity())
        if use_dcn and _HAS_DCN:
            cls_layers = [RepDWSBlock(cls_channels) for _ in range(cls_convs - 2)]
            cls_layers += [DCNv2Block(cls_channels) for _ in range(2)]
            self.cls_tower = nn.Sequential(*cls_layers)
        else:
            self.cls_tower = nn.Sequential(
                *[RepDWSBlock(cls_channels) for _ in range(cls_convs)])
        self.cls_eca = ECABlock(cls_channels) if use_eca else nn.Identity()

        # ── Regression path (kept lightweight, FPN width) ─────────────────────
        self.reg_tower = nn.Sequential(
            *[RepDWSBlock(in_channels) for _ in range(num_convs)])

        # ── Shared per-level regression heads (operate on reg-tower width) ────
        reg_out_ch = 4 * (reg_max + 1) if reg_max > 0 else 4
        self.bbox_reg = nn.ModuleList([
            nn.Conv2d(in_channels, reg_out_ch, 3, padding=1) for _ in range(num_levels)])
        if use_centerness:
            self.centerness = nn.ModuleList([
                nn.Conv2d(in_channels, 1, 3, padding=1) for _ in range(num_levels)])
        if obb:
            self.angle_reg = nn.ModuleList([
                nn.Conv2d(in_channels, 1, 3, padding=1) for _ in range(num_levels)])

        # Per-level learnable regression scales
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_levels)])

        # ── Dual per-level classifiers (operate on cls-tower width) ───────────
        def _make_cls(num_classes: int) -> nn.ModuleList:
            if cosine_cls:
                return nn.ModuleList([
                    CosineConv2d(cls_channels, num_classes,
                                 scale_init=cosine_scale, prior_prob=prior_prob)
                    for _ in range(num_levels)])
            convs = nn.ModuleList([
                nn.Conv2d(cls_channels, num_classes, 3, padding=1)
                for _ in range(num_levels)])
            bias_cls = float(-math.log((1.0 - prior_prob) / prior_prob))
            for conv in convs:
                nn.init.constant_(conv.bias, bias_cls)
                nn.init.normal_(conv.weight, std=0.01)
            return convs

        self.cls_coco   = _make_cls(num_coco_classes)
        self.cls_neubie = _make_cls(num_neubie_classes)

        # ── Initialise regression heads ───────────────────────────────────────
        init_lists = [self.bbox_reg]
        if use_centerness:
            init_lists.append(self.centerness)
        for mod_list in init_lists:
            for conv in mod_list:
                nn.init.normal_(conv.weight, std=0.001)
                nn.init.zeros_(conv.bias)
        if obb:
            for conv in self.angle_reg:
                nn.init.normal_(conv.weight, std=0.001)
                nn.init.zeros_(conv.bias)

    def forward(self, features: List[torch.Tensor],
                dataset: str = 'neubie') -> Dict[str, List[torch.Tensor]]:
        """
        features : list of FPN level tensors [P3, P4, P5]
        dataset  : 'coco' or 'neubie' — selects which cls head to use
        Returns dict with keys 'cls', 'reg', 'ctr' [, 'angle'] — same as v3.
        """
        cls_head = self.cls_coco if dataset == 'coco' else self.cls_neubie

        cls_out: List[torch.Tensor] = []
        reg_out: List[torch.Tensor] = []
        ctr_out: List[torch.Tensor] = []
        ang_out: List[torch.Tensor] = []
        dfl_out: List[torch.Tensor] = []

        for lvl, feat in enumerate(features):
            cls_f = self.cls_eca(self.cls_tower(self.cls_entry(feat)))
            reg_f = self.reg_tower(feat)

            cls_out.append(cls_head[lvl](cls_f))

            # ── Regression: DFL distribution or plain softplus LTRB ──────────
            raw_reg = self.bbox_reg[lvl](reg_f)
            if self.reg_max > 0:
                B, _, H_l, W_l = raw_reg.shape
                raw_4d = raw_reg.reshape(B, 4, self.reg_max + 1, H_l, W_l)
                proj = torch.arange(self.reg_max + 1,
                                    device=raw_reg.device, dtype=raw_reg.dtype)
                dfl_val = (F.softmax(raw_4d, dim=2) *
                           proj.reshape(1, 1, -1, 1, 1)).sum(dim=2)
                reg_out.append(dfl_val * self.scales[lvl])
                dfl_out.append(raw_reg)
            else:
                reg_out.append(
                    F.softplus(raw_reg) * self.scales[lvl])

            # ── Centerness (optional) ────────────────────────────────────────
            if self.use_centerness:
                ctr_out.append(self.centerness[lvl](reg_f))
            else:
                ctr_out.append(torch.zeros(
                    feat.shape[0], 1, feat.shape[2], feat.shape[3],
                    device=feat.device))

            if self.obb:
                ang_out.append(
                    (torch.sigmoid(self.angle_reg[lvl](reg_f)) - 0.25) * math.pi)

        out: Dict[str, List[torch.Tensor]] = {
            'cls': cls_out, 'reg': reg_out, 'ctr': ctr_out}
        if self.obb:
            out['angle'] = ang_out
        if dfl_out:
            out['dfl_raw'] = dfl_out
        return out

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ──────────────────────────────────────────────────────────────────────────────
# Top-Level Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DINODetectionHeadMix(nn.Module):
    """Multi-layer DINOv3 features → MultiLevelEfficientPAN → DetectHeadMix.

    Parameters
    ----------
    backbone_out_channels : embed_dim of frozen DINOv3 (384 for ViT-S+)
    fpn_channels          : FPN internal width (192 default)
    num_coco_classes      : 80 for standard COCO
    num_neubie_classes    : fixed Neubie classes (16)
    num_convs             : regression tower depth (4 default)
    cls_convs             : classification tower depth (6 default — decoupled)
    cls_channels          : classification tower width (256 default — decoupled)
    obb                   : enable OBB angle branch
    use_eca               : ECA channel attention at cls-tower output
    cosine_cls            : cosine (normalized) final classifier (long-tail)
    cosine_scale          : initial learnable temperature for the cosine head
    """

    def __init__(self, backbone_out_channels: int = 384,
                 fpn_channels: int = 192,
                 num_coco_classes: int = 80,
                 num_neubie_classes: int = 16,
                 num_convs: int = 4,
                 cls_convs: int = 6,
                 cls_channels: int = 256,
                 obb: bool = True,
                 use_eca: bool = True,
                 cosine_cls: bool = True,
                 cosine_scale: float = 20.0):
        super().__init__()
        self.fpn  = MultiLevelEfficientPAN(backbone_out_channels,
                                           out_channels=fpn_channels)
        self.head = DetectHeadMix(
            in_channels=fpn_channels,
            num_coco_classes=num_coco_classes,
            num_neubie_classes=num_neubie_classes,
            num_convs=num_convs,
            cls_convs=cls_convs,
            cls_channels=cls_channels,
            num_levels=3,
            obb=obb,
            use_eca=use_eca,
            cosine_cls=cosine_cls,
            cosine_scale=cosine_scale,
        )

    def forward(self, features: List[torch.Tensor],
                dataset: str = 'neubie') -> Dict:
        """features: [shallow, mid, deep] from DinoBackboneV3"""
        return self.head(self.fpn(features), dataset=dataset)

    def fuse(self):
        self.fpn.fuse()
        self.head.fuse()
        return self

    def shared_params(self):
        """All parameters except the two cls heads (for per-component LR).

        The shared cls feature extractor (cls_entry / cls_tower / cls_eca) and
        the reg path stay here; only the dataset-specific cosine/plain
        classifiers (cls_coco, cls_neubie) are excluded.
        """
        skip_ids = {id(p) for m in (self.head.cls_coco, self.head.cls_neubie)
                    for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in skip_ids]

    def coco_cls_params(self):
        return list(self.head.cls_coco.parameters())

    def neubie_cls_params(self):
        return list(self.head.cls_neubie.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed  = 384   # ViT-S+

    H_feat, W_feat = 30, 40
    feats = [torch.randn(1, embed, H_feat, W_feat, device=device) for _ in range(3)]

    head = DINODetectionHeadMix(
        backbone_out_channels=embed,
        fpn_channels=192,
        num_coco_classes=80,
        num_neubie_classes=16,
        num_convs=4,
        cls_convs=6,
        cls_channels=256,
        obb=False,
        use_eca=True,
        cosine_cls=True,
    ).to(device)

    out_coco   = head(feats, dataset='coco')
    out_neubie = head(feats, dataset='neubie')

    print('COCO   cls shapes:', [tuple(t.shape) for t in out_coco['cls']])
    print('Neubie cls shapes:', [tuple(t.shape) for t in out_neubie['cls']])
    print('Shared reg shapes:', [tuple(t.shape) for t in out_coco['reg']])

    params = count_parameters(head)
    print(f'Total params : {params["total"]/1e6:.3f} M  '
          f'(trainable: {params["trainable"]/1e6:.3f} M)')

    shared = sum(p.numel() for p in head.shared_params())
    coco   = sum(p.numel() for p in head.coco_cls_params())
    neubie = sum(p.numel() for p in head.neubie_cls_params())
    print(f'  Shared     : {shared/1e6:.3f} M')
    print(f'  COCO   cls : {coco/1e6:.3f} M')
    print(f'  Neubie cls : {neubie/1e6:.3f} M')
