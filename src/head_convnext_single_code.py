"""
model_head_convnext.py — ConvNeXt-backbone detection head (mix_p2 strategy)
===========================================================================

Standalone, self-contained detection head for ConvNeXt backbones. Architecturally
parallel to the ViT `mix_p2` head — same output format, same loss/decode/optimizer
compatibility, but a different FPN that exploits ConvNeXt's naturally multi-scale
features.

Key difference vs ViT FPN (MultiLevelEfficientPANP2):
    ViT:      3 feature maps at SAME resolution (H/16, W/16) and SAME channels.
              FPN must artificially create multi-scale via strided convolutions.
    ConvNeXt: 3 feature maps at DIFFERENT resolutions AND different channels.
              FPN uses per-stage projection layers; no artificial downsampling
              needed for P2/P3/P4 — only P5 (stride 64) is synthesized.

Pyramid mapping (480x640 input):
    Stage 1 (stride  8, C1) -> P2 (60x80)   -- no downsample needed
    Stage 2 (stride 16, C2) -> P3 (30x40)   -- no downsample needed
    Stage 3 (stride 32, C3) -> P4 (15x20)   -- no downsample needed
    P5 = downsample(P4, 2x) -> P5 (8x10)    -- only artificial level

Output: [P2, P3, P4, P5] at strides [8, 16, 32, 64] — same as mix_p2.

This module is fully self-contained: it does NOT import from model_head_mix.py
or model_head_mix_p2.py. The `DetectHeadMix`, `AIFIEncoder`, and `AuxDecoderHead`
building blocks (and their private helpers `DCNv2Block`, `CosineConv2d`,
`CrossLevelAttention`) are inlined below, copied verbatim from those two files.
It still imports shared low-level building blocks (ConvGNReLU, DWSConvGNReLU,
RepDWSBlock, SmallObjectRefine, ECABlock, count_parameters) from model_head_v2.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_head_v2 import (
    ConvGNReLU,
    DWSConvGNReLU,
    RepDWSBlock,
    SmallObjectRefine,
    ECABlock,
    count_parameters,
)

try:
    from torchvision.ops import DeformConv2d
    _HAS_DCN = True
except ImportError:
    _HAS_DCN = False


# ══════════════════════════════════════════════════════════════════════════════
# Inlined from model_head_mix.py — dual-classifier detection head building blocks
# ══════════════════════════════════════════════════════════════════════════════

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
# Cross-Level Attention — lightweight feature interaction across FPN levels
# ──────────────────────────────────────────────────────────────────────────────

class CrossLevelAttention(nn.Module):
    """Lightweight cross-level feature interaction for the cls tower.

    Pools cls features from all FPN levels into a single token sequence,
    applies multi-head self-attention with per-level embeddings, then
    scatters back to per-level feature maps.

    For 480x640 input with 4 FPN levels:
        P2(60x80) + P3(30x40) + P4(15x20) + P5(8x10) = 6380 tokens.
        With 1 layer at dim=256, heads=8, this adds ~2M params / ~5ms.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 1,
                 max_levels: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.level_embed = nn.Parameter(torch.randn(max_levels, dim) * 0.02)

    def forward(self, cls_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        cls_feats: list of (B, C, H_l, W_l) cls tower outputs per level.
        Returns: list of (B, C, H_l, W_l) attention-enhanced features.
        """
        B, C = cls_feats[0].shape[:2]
        tokens_per_level = []
        sizes = []
        for lvl, feat in enumerate(cls_feats):
            H, W = feat.shape[2], feat.shape[3]
            sizes.append((H, W))
            t = feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
            t = t + self.level_embed[lvl].unsqueeze(0).unsqueeze(0)
            tokens_per_level.append(t)

        all_tokens = torch.cat(tokens_per_level, dim=1)  # (B, sum(H*W), C)
        all_tokens = self.norm(self.encoder(all_tokens))

        out = []
        offset = 0
        for lvl, (H, W) in enumerate(sizes):
            n = H * W
            t = all_tokens[:, offset:offset + n]  # (B, H*W, C)
            out.append(t.permute(0, 2, 1).reshape(B, C, H, W))
            offset += n
        return out


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
                 use_dcn: bool = False,
                 use_cross_level_attn: bool = False):
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

        # ── Cross-level attention (optional) ──────────────────────────────────
        self.cross_level_attn = (
            CrossLevelAttention(cls_channels, num_heads=8, num_layers=1)
            if use_cross_level_attn else None
        )

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
                dataset: str = 'neubie',
                return_cls_feat: bool = False) -> Dict[str, List[torch.Tensor]]:
        """
        features : list of FPN level tensors [P3, P4, P5] (or [P2, P3, P4, P5])
        dataset  : 'coco' or 'neubie' — selects which cls head to use
        return_cls_feat : if True, include 'cls_feat' in output (for CPD loss)
        Returns dict with keys 'cls', 'reg', 'ctr' [, 'angle'] — same as v3.
        """
        cls_head = self.cls_coco if dataset == 'coco' else self.cls_neubie

        cls_out: List[torch.Tensor] = []
        cls_feat_out: List[torch.Tensor] = []
        reg_out: List[torch.Tensor] = []
        ctr_out: List[torch.Tensor] = []
        ang_out: List[torch.Tensor] = []
        dfl_out: List[torch.Tensor] = []

        # ── Cls tower: collect all levels, then optionally apply cross-level attn
        cls_feats = [self.cls_eca(self.cls_tower(self.cls_entry(feat)))
                     for feat in features]
        if self.cross_level_attn is not None:
            cls_feats = self.cross_level_attn(cls_feats)

        for lvl, feat in enumerate(features):
            cls_f = cls_feats[lvl]
            reg_f = self.reg_tower(feat)

            cls_out.append(cls_head[lvl](cls_f))
            if return_cls_feat:
                cls_feat_out.append(cls_f)

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
        if return_cls_feat:
            out['cls_feat'] = cls_feat_out
        return out

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ══════════════════════════════════════════════════════════════════════════════
# Inlined from model_head_mix_p2.py — AIFI encoder + auxiliary decoder head
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# AIFI Encoder: multi-head self-attention on the coarsest feature map (P5)
# ──────────────────────────────────────────────────────────────────────────────

class AIFIEncoder(nn.Module):
    """Attention-based Intra-scale Feature Interaction on the coarsest FPN level.

    Applies multi-head self-attention to P5 (8×10 = 80 tokens for 480×640 input).
    Uses 2D sinusoidal positional encoding so the model knows spatial layout.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self._pos_cache: Optional[torch.Tensor] = None
        self._pos_hw = (0, 0)

    def _get_pos_embed(self, H: int, W: int, C: int,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """2D sinusoidal positional encoding (cached)."""
        if self._pos_cache is not None and self._pos_hw == (H, W):
            return self._pos_cache.to(device=device, dtype=dtype)
        half = C // 2
        gy = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        gx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
        dim_t = torch.arange(half // 2, device=device, dtype=torch.float32)
        dim_t = 10000.0 ** (2.0 * dim_t / half)
        # y-axis encoding (first quarter of channels)
        pe_y = torch.zeros(H, W, half, device=device)
        pe_y[..., 0::2] = torch.sin(gy.unsqueeze(-1) / dim_t)
        pe_y[..., 1::2] = torch.cos(gy.unsqueeze(-1) / dim_t)
        # x-axis encoding (second quarter)
        pe_x = torch.zeros(H, W, half, device=device)
        pe_x[..., 0::2] = torch.sin(gx.unsqueeze(-1) / dim_t)
        pe_x[..., 1::2] = torch.cos(gx.unsqueeze(-1) / dim_t)
        pos = torch.cat([pe_y, pe_x], dim=-1).reshape(H * W, C).to(dtype)
        self._pos_cache = pos
        self._pos_hw = (H, W)
        return pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pos = self._get_pos_embed(H, W, C, x.device, x.dtype)
        tokens = x.flatten(2).permute(0, 2, 1)       # (B, H*W, C)
        tokens = tokens + pos.unsqueeze(0)
        tokens = self.norm(self.encoder(tokens))
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary Decoder: lightweight transformer decoder for training supervision
# ──────────────────────────────────────────────────────────────────────────────

class AuxDecoderHead(nn.Module):
    """Lightweight transformer decoder for auxiliary training-only supervision.

    Takes multi-level FPN features as encoder memory, uses learned object queries,
    and produces box + cls predictions via Hungarian matching loss. Discarded at
    inference (controlled by model.training flag).
    """

    def __init__(self, dim: int = 256, num_queries: int = 100,
                 num_classes: int = 16, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Embedding(num_queries, dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(dim, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True),
            nn.Linear(dim, 4),                         # cxcywh normalized
        )
        # Init cls bias for focal-style balance
        prior_prob = 0.01
        nn.init.constant_(self.cls_head.bias,
                          -math.log((1.0 - prior_prob) / prior_prob))
        nn.init.normal_(self.cls_head.weight, std=0.01)

    def forward(self, fpn_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = fpn_features[0].shape[0]
        memory_list = [feat.flatten(2).permute(0, 2, 1) for feat in fpn_features]
        memory = torch.cat(memory_list, dim=1)           # (B, N_total, C)

        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)               # (B, Q, C)

        return {
            'aux_cls':   self.cls_head(out),              # (B, Q, num_classes)
            'aux_boxes': self.reg_head(out).sigmoid(),    # (B, Q, 4) normalized cxcywh
        }


# ══════════════════════════════════════════════════════════════════════════════
# ConvNeXt-specific modules (unchanged from the original model_head_convnext.py)
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# FreqEnhance: Difference-of-Gaussians frequency band enhancement
# ──────────────────────────────────────────────────────────────────────────────

class FreqEnhance(nn.Module):
    """Per-level frequency-band feature enhancement via Difference of Gaussians.

    Decomposes each FPN feature map into three frequency bands (low / mid / high),
    processes each through a lightweight depthwise conv, and fuses them with
    learnable scalar weights + a residual connection.

    Inspired by SFDNet (ECCV 2026) — simplified: no Mamba SSM, just DoG + DWConv.

    Parameters
    ----------
    channels : FPN channel count
    sigma_low  : Gaussian sigma for the low-pass filter (larger = smoother)
    sigma_high : Gaussian sigma for the high-pass boundary (smaller = finer)
    kernel_size: Gaussian blur kernel size (must be odd)
    """

    def __init__(self, channels: int, sigma_low: float = 2.0,
                 sigma_high: float = 1.0, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        self.channels = channels

        # Fixed Gaussian kernels (not learned — band boundaries are stable)
        self.register_buffer('_gauss_low',
                             self._make_gauss_kernel(channels, kernel_size, sigma_low))
        self.register_buffer('_gauss_high',
                             self._make_gauss_kernel(channels, kernel_size, sigma_high))
        self.pad = kernel_size // 2

        # Per-band 3×3 depthwise conv (cheap, learns band-specific refinement)
        self.conv_low  = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.conv_mid  = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.conv_high = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)

        # Learnable per-band scalar weights (init: equal contribution)
        self.alpha = nn.Parameter(torch.tensor(1.0 / 3))  # low
        self.beta  = nn.Parameter(torch.tensor(1.0 / 3))  # mid
        self.gamma = nn.Parameter(torch.tensor(1.0 / 3))  # high

        # GroupNorm after fusion
        g = min(16, channels)
        while channels % g != 0 and g > 1:
            g -= 1
        self.norm = nn.GroupNorm(g, channels)

    @staticmethod
    def _make_gauss_kernel(channels: int, ks: int, sigma: float) -> torch.Tensor:
        """Create a grouped 2D Gaussian blur kernel (one kernel per channel)."""
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)  # (ks, ks)
        # Grouped conv kernel: (C_out, C_in/groups, kH, kW) with groups=C
        return g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, ks, ks).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian blurs (depthwise grouped conv with fixed kernels)
        x_blur_low  = F.conv2d(x, self._gauss_low,  padding=self.pad, groups=self.channels)
        x_blur_high = F.conv2d(x, self._gauss_high, padding=self.pad, groups=self.channels)

        # Frequency band decomposition
        p_low  = x_blur_low                     # smooth / large-scale
        p_mid  = x_blur_high - x_blur_low       # mid-frequency
        p_high = x - x_blur_high                # fine detail / edges

        # Per-band refinement + weighted fusion + residual
        fused = (self.alpha * self.conv_low(p_low) +
                 self.beta  * self.conv_mid(p_mid) +
                 self.gamma * self.conv_high(p_high))
        return self.norm(fused) + x


# ──────────────────────────────────────────────────────────────────────────────
# ConvNeXt FPN: multi-scale inputs with per-stage channel projection
# ──────────────────────────────────────────────────────────────────────────────

class ConvNeXtEfficientPAN(nn.Module):
    """Bidirectional FPN for ConvNeXt multi-scale features.

    Unlike MultiLevelEfficientPANP2 (ViT), this FPN receives features at
    DIFFERENT resolutions and channel counts. No artificial downsampling is
    needed for P2/P3/P4 — only P5 is synthesized by downsampling P4.

    Inputs
    ------
    features : [stage1, stage2, stage3] from DinoBackboneConvNeXt
        stage1: (B, C1, H/8,  W/8)   -- stride 8
        stage2: (B, C2, H/16, W/16)  -- stride 16
        stage3: (B, C3, H/32, W/32)  -- stride 32

    Build
    -----
    Stage 1 -- project each input to FPN channels (different input C per stage):
        P2 = proj_s8(stage1)      stride 8,  60x80   (no spatial change)
        P3 = proj_s16(stage2)     stride 16, 30x40   (no spatial change)
        P4 = proj_s32(stage3)     stride 32, 15x20   (no spatial change)
        P5 = down_p5(P4)          stride 64, 8x10    (only artificial level)

    Stage 2 -- AIFI on P5 (optional):
        P5 = aifi(P5)

    Stage 3 -- top-down (coarse -> fine):
        P5_td = smooth(P5)
        P4_td = smooth(P4 + upsample(P5_td))
        P3_td = smooth(P3 + upsample(P4_td))
        P2    = p2_refine(P2 + upsample(P3_td))  -- SmallObjectRefine

    Stage 4 -- bottom-up (fine -> coarse):
        P3_bu = P3_td + down(P2)
        P4_bu = smooth(P4_td + down(P3_bu))
        P5_bu = smooth(P5_td + down(P4_bu))

    Output: [P2, P3_bu, P4_bu, P5_bu]  (finest-first)
    """

    def __init__(self, in_channels_list: List[int],
                 out_channels: int = 256,
                 use_aifi: bool = False,
                 aifi_layers: int = 2,
                 aifi_heads: int = 8):
        super().__init__()
        assert len(in_channels_list) == 3, \
            f'Expected 3 input channel counts [C1, C2, C3], got {len(in_channels_list)}'
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

        # Per-stage projection (different input channels each)
        self.proj_s8  = _proj(in_channels_list[0], C)   # stage1 -> P2
        self.proj_s16 = _proj(in_channels_list[1], C)   # stage2 -> P3
        self.proj_s32 = _proj(in_channels_list[2], C)   # stage3 -> P4

        # P5: downsample P4 by 2x (only artificial level)
        self.down_p5 = DWSConvGNReLU(C, stride=2)

        # AIFI (optional): self-attention on coarsest level
        self.use_aifi = use_aifi
        if use_aifi:
            self.aifi = AIFIEncoder(C, num_heads=aifi_heads,
                                    num_layers=aifi_layers)

        # Top-down smoothing
        self.td_smooth5 = RepDWSBlock(C)
        self.td_smooth4 = RepDWSBlock(C)
        self.td_smooth3 = RepDWSBlock(C)

        # P2 refinement (small-object)
        self.p2_refine = SmallObjectRefine(C)

        # Bottom-up path
        self.bu_down_p3 = DWSConvGNReLU(C, stride=2)   # P2 -> P3
        self.bu_down_p4 = DWSConvGNReLU(C, stride=2)   # P3 -> P4
        self.bu_smooth4 = RepDWSBlock(C)
        self.bu_down_p5 = DWSConvGNReLU(C, stride=2)   # P4 -> P5
        self.bu_smooth5 = RepDWSBlock(C)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        stage1, stage2, stage3 = features

        # ---- Project to FPN channels (no spatial change for P2/P3/P4) ----
        p2_raw = self.proj_s8(stage1)     # stride 8
        p3_raw = self.proj_s16(stage2)    # stride 16
        p4_raw = self.proj_s32(stage3)    # stride 32
        p5_raw = self.down_p5(p4_raw)     # stride 64 (only artificial level)

        # ---- AIFI on coarsest level ----
        if self.use_aifi:
            p5_raw = self.aifi(p5_raw)

        # ---- Top-down path ----
        p5_td = self.td_smooth5(p5_raw)
        p4_td = self.td_smooth4(
            p4_raw + F.interpolate(p5_td, size=p4_raw.shape[-2:], mode='nearest'))
        p3_td = self.td_smooth3(
            p3_raw + F.interpolate(p4_td, size=p3_raw.shape[-2:], mode='nearest'))

        # P2: inject top-down P3 semantics + small-object refine
        p2 = self.p2_refine(
            p2_raw + F.interpolate(p3_td, size=p2_raw.shape[-2:], mode='nearest'))

        # ---- Bottom-up path ----
        p3_bu = p3_td + self.bu_down_p3(p2)
        p4_bu = self.bu_smooth4(p4_td + self.bu_down_p4(p3_bu))
        p5_bu = self.bu_smooth5(p5_td + self.bu_down_p5(p4_bu))

        return [p2, p3_bu, p4_bu, p5_bu]   # finest-first

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ──────────────────────────────────────────────────────────────────────────────
# Top-level wrapper: ConvNeXt FPN + DetectHeadMix (4-level)
# ──────────────────────────────────────────────────────────────────────────────

class ConvNeXtDetectionHeadMixP2(nn.Module):
    """ConvNeXt multi-scale features -> ConvNeXtEfficientPAN -> 4-level DetectHeadMix.

    Same output format as DINODetectionHeadMixP2 (ViT):
        {'cls': [...], 'reg': [...], 'ctr': [...], 'dfl_raw': [...], 'aux': {...}}
    so loss_v3.py, utils_OBB.py, and the training loop work unchanged.

    Parameters
    ----------
    in_channels_list : channel counts for stages 1, 2, 3 of ConvNeXt
        small: [192, 384, 768]
        base:  [256, 512, 1024]
    fpn_channels    : FPN internal width (256 default for Phase C)
    Phase C knobs   : AIFI, DFL, no centerness, DCNv2, aux decoder
    """

    def __init__(self, in_channels_list: List[int],
                 fpn_channels: int = 256,
                 num_coco_classes: int = 80,
                 num_neubie_classes: int = 16,
                 num_convs: int = 4,
                 cls_convs: int = 6,
                 cls_channels: int = 256,
                 obb: bool = False,
                 use_eca: bool = True,
                 cosine_cls: bool = True,
                 cosine_scale: float = 20.0,
                 # Phase C knobs
                 use_aifi: bool = False,
                 aifi_layers: int = 2,
                 aifi_heads: int = 8,
                 reg_max: int = 0,
                 use_centerness: bool = True,
                 use_dcn: bool = False,
                 use_aux_decoder: bool = False,
                 aux_num_queries: int = 100,
                 aux_decoder_layers: int = 2,
                 use_freq_enhance: bool = False,
                 use_cross_level_attn: bool = False):
        super().__init__()
        self.num_levels = 4
        self.fpn = ConvNeXtEfficientPAN(
            in_channels_list, out_channels=fpn_channels,
            use_aifi=use_aifi, aifi_layers=aifi_layers, aifi_heads=aifi_heads)

        # DoG frequency-band enhancement on each FPN level (SFDNet-inspired)
        self.use_freq_enhance = use_freq_enhance
        if use_freq_enhance:
            self.freq_enhance = nn.ModuleList([
                FreqEnhance(fpn_channels) for _ in range(4)])

        self.head = DetectHeadMix(
            in_channels=fpn_channels,
            num_coco_classes=num_coco_classes,
            num_neubie_classes=num_neubie_classes,
            num_convs=num_convs,
            cls_convs=cls_convs,
            cls_channels=cls_channels,
            num_levels=4,
            obb=obb,
            use_eca=use_eca,
            cosine_cls=cosine_cls,
            cosine_scale=cosine_scale,
            reg_max=reg_max,
            use_centerness=use_centerness,
            use_dcn=use_dcn,
            use_cross_level_attn=use_cross_level_attn,
        )
        self.use_aux_decoder = use_aux_decoder
        if use_aux_decoder:
            self.aux_decoder = AuxDecoderHead(
                dim=fpn_channels, num_queries=aux_num_queries,
                num_classes=num_neubie_classes, num_layers=aux_decoder_layers,
            )

    def forward(self, features: List[torch.Tensor],
                dataset: str = 'neubie',
                return_cls_feat: bool = False) -> Dict:
        fpn_out = self.fpn(features)
        if self.use_freq_enhance:
            fpn_out = [self.freq_enhance[i](f) for i, f in enumerate(fpn_out)]
        head_out = self.head(fpn_out, dataset=dataset,
                             return_cls_feat=return_cls_feat)
        if self.training and self.use_aux_decoder:
            head_out['aux'] = self.aux_decoder(fpn_out)
        return head_out

    def fuse(self):
        self.fpn.fuse()
        self.head.fuse()
        return self

    # ── Param-group helpers (identical partitioning to DINODetectionHeadMixP2) ──
    def shared_params(self):
        skip_ids = {id(p) for m in (self.head.cls_coco, self.head.cls_neubie)
                    for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in skip_ids]

    def coco_cls_params(self):
        return list(self.head.cls_coco.parameters())

    def neubie_cls_params(self):
        return list(self.head.cls_neubie.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test with ConvNeXt small channel dims
    in_ch = [192, 384, 768]
    model = ConvNeXtDetectionHeadMixP2(
        in_channels_list=in_ch,
        fpn_channels=256,
        num_coco_classes=80,
        num_neubie_classes=16,
        num_convs=4, cls_convs=6, cls_channels=256,
        obb=False, use_eca=True, cosine_cls=True,
        use_aifi=False,
        reg_max=16,
        use_centerness=False,
        use_dcn=True,
        use_aux_decoder=True,
    ).to(device)

    # Simulate ConvNeXt small backbone outputs for 480x640 input
    feats = [
        torch.randn(2, 192, 60, 80, device=device),    # stage1: stride 8
        torch.randn(2, 384, 30, 40, device=device),    # stage2: stride 16
        torch.randn(2, 768, 15, 20, device=device),    # stage3: stride 32
    ]

    # Training mode (aux decoder active)
    model.train()
    out = model(feats, dataset='neubie')
    print('Output keys:', list(out.keys()))
    print('levels:', len(out['cls']))
    print('cls shapes:', [tuple(t.shape) for t in out['cls']])
    print('reg shapes:', [tuple(t.shape) for t in out['reg']])
    if 'dfl_raw' in out:
        print('dfl_raw shapes:', [tuple(t.shape) for t in out['dfl_raw']])
    if 'aux' in out:
        print('aux_cls:', tuple(out['aux']['aux_cls'].shape))
        print('aux_boxes:', tuple(out['aux']['aux_boxes'].shape))

    # Expected: P2(60x80), P3(30x40), P4(15x20), P5(8x10)
    expected_sizes = [(60, 80), (30, 40), (15, 20), (8, 10)]
    for i, (h, w) in enumerate(expected_sizes):
        actual_h, actual_w = out['cls'][i].shape[2], out['cls'][i].shape[3]
        status = 'OK' if (actual_h, actual_w) == (h, w) else 'MISMATCH'
        print(f'  P{i+2}: expected ({h},{w}), got ({actual_h},{actual_w}) [{status}]')

    # Eval mode (no aux decoder)
    model.eval()
    with torch.no_grad():
        out_eval = model(feats, dataset='neubie')
    print('\nEval mode keys:', list(out_eval.keys()))
    assert 'aux' not in out_eval, 'aux should not be in eval output'

    # Backward pass test
    model.train()
    out = model(feats, dataset='neubie')
    loss = sum(t.sum() for t in out['cls']) + sum(t.sum() for t in out['reg'])
    loss.backward()
    print('\nBackward pass: OK')

    # Param counts
    p = count_parameters(model)
    print(f'\nParams total: {p["total"]/1e6:.3f} M  '
          f'(trainable: {p["trainable"]/1e6:.3f} M)')
    shared = sum(pp.numel() for pp in model.shared_params())
    coco = sum(pp.numel() for pp in model.coco_cls_params())
    neubie = sum(pp.numel() for pp in model.neubie_cls_params())
    print(f'  Shared:     {shared/1e6:.3f} M')
    print(f'  COCO cls:   {coco/1e6:.3f} M')
    print(f'  Neubie cls: {neubie/1e6:.3f} M')

    # Test with ConvNeXt base channel dims
    print('\n--- ConvNeXt base ---')
    in_ch_base = [256, 512, 1024]
    model_base = ConvNeXtDetectionHeadMixP2(
        in_channels_list=in_ch_base,
        fpn_channels=256,
        num_coco_classes=80, num_neubie_classes=16,
        num_convs=4, cls_convs=6, cls_channels=256,
        obb=False, use_eca=True, cosine_cls=True,
        reg_max=16, use_centerness=False, use_dcn=True,
        use_aux_decoder=True,
    ).to(device)

    feats_base = [
        torch.randn(2, 256, 60, 80, device=device),
        torch.randn(2, 512, 30, 40, device=device),
        torch.randn(2, 1024, 15, 20, device=device),
    ]
    model_base.train()
    out_base = model_base(feats_base, dataset='neubie')
    print('cls shapes:', [tuple(t.shape) for t in out_base['cls']])

    p_base = count_parameters(model_base)
    print(f'Params total: {p_base["total"]/1e6:.3f} M')
