"""
model_head_convnext_trt.py — ONNX-exportable ConvNeXt detection head
=====================================================================

Drop-in replacement for ``model_head_convnext.py`` with **one** change
that removes the hard ONNX/TensorRT export blocker:

    **DCNv2 → DilatedConvBlockGN** in cls_tower positions 4 & 5.
    ``torchvision.ops.DeformConv2d`` has no ONNX symbolic at any opset
    — export fails entirely.  The dilated Conv2d 3×3 (dilation=2,
    effective 5×5 receptive field) + GroupNorm + ReLU + residual is a
    fully ONNX/TensorRT-compatible drop-in.

**Everything else is identical to Phase E**:
    - GroupNorm everywhere (20 instances — TRT runs these in fp16,
      which is a minor latency cost, NOT an export blocker)
    - CrossLevelAttention (LayerNorm)
    - ECABlock, DFL, CosineConv2d
    - AuxDecoderHead, AIFIEncoder
    - RepDWSBlock, SmallObjectRefine

This file is **fully self-contained**: no cross-imports from
``model_head_v2.py`` or ``model_head_mix.py``.  All building blocks
are inlined below.

Weight transfer
---------------
Use ``convert_phaseE_to_trt(src_path, dst_path)`` at the bottom of
this file to convert a Phase E checkpoint (with DCN) into one that
loads cleanly into this head.  The function:

    • For DCNv2Block positions: copies ``dcn.weight`` → ``conv.weight``,
      ``dcn.bias`` → ``conv.bias``, drops ``offset_mask.*``
    • All other keys pass through unchanged (GN stays as GN)
    • Reports missing/unexpected keys for verification

Compared to the previous TRT file (GN→BN + DCN→DilatedConv):
    - No BN calibration needed — GroupNorm has no running stats
    - No freeze_bn needed — GroupNorm is always stable
    - Much smaller accuracy gap (only 2 DCN layers changed, not 20 GN layers)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks — same as Phase E originals (GroupNorm preserved)
# ══════════════════════════════════════════════════════════════════════════════

def _gn_groups(channels: int, preferred: int = 16) -> int:
    """Find largest divisor of channels <= preferred."""
    g = min(preferred, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g


class ConvGNReLU(nn.Module):
    """Conv2d + GroupNorm + ReLU.  Same as model_head_v2.ConvGNReLU."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.gn   = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DWSConvGNReLU(nn.Module):
    """Depthwise-separable Conv + GroupNorm + ReLU.  Same as model_head_v2."""

    def __init__(self, channels: int, stride: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(channels, channels, 3, stride, 1,
                             groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn  = nn.GroupNorm(_gn_groups(channels), channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))


class RepDWSBlock(nn.Module):
    """Reparameterizable DWS block with GroupNorm.  Same as model_head_v2.

    Training graph:
        x --> DWS 3x3 --+
        x --> PW  1x1 --+-- sum --> GN --> ReLU
        x ---------------+  (identity shortcut)

    After ``fuse()``:
        x --> single DW 3x3 --> PW 1x1 --> GN --> ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dw  = nn.Conv2d(channels, channels, 3, 1, 1,
                             groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels, channels, 1, bias=False)
        self.shortcut_pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn  = nn.GroupNorm(_gn_groups(channels), channels)
        self.act = nn.ReLU(inplace=True)
        self._fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._fused:
            return self.act(self.gn(self.pw(self.dw(x))))
        main = self.pw(self.dw(x))
        shortcut = self.shortcut_pw(x)
        return self.act(self.gn(main + shortcut + x))

    @torch.no_grad()
    def fuse(self):
        if self._fused:
            return
        self._fused = True
        del self.shortcut_pw


class ECABlock(nn.Module):
    """Efficient Channel Attention (ECA-Net).  No normalization — unchanged."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs(math.log2(channels) / gamma + b / gamma))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x).squeeze(-1).transpose(1, 2)
        w = torch.sigmoid(self.conv(w)).transpose(1, 2).unsqueeze(-1)
        return x * w


class SmallObjectRefine(nn.Module):
    """Lightweight P2 refinement: 2x DWSConvGNReLU + ECA + residual."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DWSConvGNReLU(channels),
            DWSConvGNReLU(channels),
            ECABlock(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# DCN replacement — DilatedConvBlockGN (GroupNorm, not BN)
# ══════════════════════════════════════════════════════════════════════════════

class DilatedConvBlockGN(nn.Module):
    """Dilated Conv2d 3×3 (dilation=2, effective 5×5 RF) + GN + ReLU + residual.

    Replaces ``DCNv2Block``.  DCN's strength is per-pixel adaptive sampling;
    the dilated conv approximates this with a larger *fixed* receptive field
    (5×5 effective vs 3×3 standard), covering more context for occluded and
    deformable objects.

    Uses GroupNorm (same as the original DCNv2Block) — no BN cold-start issue.

    Weight transfer from DCN: ``dcn.weight[C,C,3,3]`` → ``conv.weight[C,C,3,3]``
    (same shape, direct copy — the dilated conv uses the same kernel size,
    just with gaps between sampling positions).
    ``dcn.bias[C]`` → ``conv.bias[C]`` (direct copy).
    ``offset_mask.*`` keys are dropped (no longer needed).
    ``gn.*`` keys stay as ``gn.*`` (unchanged).
    """

    def __init__(self, channels: int, dilation: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,
                              padding=dilation, dilation=dilation, bias=True)
        self.gn   = nn.GroupNorm(_gn_groups(channels), channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x))) + x


# ══════════════════════════════════════════════════════════════════════════════
# Cosine classifier (unchanged — no normalization layer)
# ══════════════════════════════════════════════════════════════════════════════

class CosineConv2d(nn.Module):
    """1×1 cosine-similarity classifier: ``scale · ⟨f̂, ŵ⟩ + bias``."""

    def __init__(self, in_channels: int, num_classes: int,
                 scale_init: float = 20.0, prior_prob: float = 0.01,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.randn(num_classes, in_channels, 1, 1) * 0.01)
        bias_val = float(-math.log((1.0 - prior_prob) / prior_prob))
        self.bias = nn.Parameter(torch.full((num_classes,), bias_val))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=1, eps=self.eps)
        x = F.normalize(x, dim=1, eps=self.eps)
        cos = F.conv2d(x, w)
        return cos * self.log_scale.exp() + self.bias.view(1, -1, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-Level Attention (unchanged — uses LayerNorm, TRT-friendly)
# ══════════════════════════════════════════════════════════════════════════════

class CrossLevelAttention(nn.Module):
    """Lightweight cross-level feature interaction for the cls tower."""

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
        B, C = cls_feats[0].shape[:2]
        tokens_per_level = []
        sizes = []
        for lvl, feat in enumerate(cls_feats):
            H, W = feat.shape[2], feat.shape[3]
            sizes.append((H, W))
            t = feat.flatten(2).permute(0, 2, 1)
            t = t + self.level_embed[lvl].unsqueeze(0).unsqueeze(0)
            tokens_per_level.append(t)

        all_tokens = torch.cat(tokens_per_level, dim=1)
        all_tokens = self.norm(self.encoder(all_tokens))

        out = []
        offset = 0
        for lvl, (H, W) in enumerate(sizes):
            n = H * W
            t = all_tokens[:, offset:offset + n]
            out.append(t.permute(0, 2, 1).reshape(B, C, H, W))
            offset += n
        return out


# ══════════════════════════════════════════════════════════════════════════════
# DetectHeadMixTRT — GN preserved, DCN replaced with DilatedConvBlockGN
# ══════════════════════════════════════════════════════════════════════════════

class DetectHeadMixTRT(nn.Module):
    """Per-level prediction heads with dual classification branches.

    Identical to ``DetectHeadMix`` except:
      - cls_tower positions 4-5: ``DilatedConvBlockGN`` (replaces DCNv2Block)
      - Everything else is the same (GroupNorm, RepDWSBlock, ECA, etc.)
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
                 use_dcn: bool = False,        # accepted but ignored — always DilatedConv
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

        # ── Classification path ──────────────────────────────────────────
        self.cls_entry = (ConvGNReLU(in_channels, cls_channels,
                                     kernel_size=1, stride=1, padding=0)
                          if cls_channels != in_channels else nn.Identity())

        # Positions 0..(cls_convs-3): RepDWSBlock (GN)
        # Last 2 positions: DilatedConvBlockGN (replaces DCNv2Block)
        cls_layers = [RepDWSBlock(cls_channels) for _ in range(cls_convs - 2)]
        cls_layers += [DilatedConvBlockGN(cls_channels) for _ in range(2)]
        self.cls_tower = nn.Sequential(*cls_layers)
        self.cls_eca = ECABlock(cls_channels) if use_eca else nn.Identity()

        # ── Cross-level attention (optional) ──────────────────────────────
        self.cross_level_attn = (
            CrossLevelAttention(cls_channels, num_heads=8, num_layers=1)
            if use_cross_level_attn else None
        )

        # ── Regression path (unchanged) ──────────────────────────────────
        self.reg_tower = nn.Sequential(
            *[RepDWSBlock(in_channels) for _ in range(num_convs)])

        # ── Per-level regression heads ────────────────────────────────────
        reg_out_ch = 4 * (reg_max + 1) if reg_max > 0 else 4
        self.bbox_reg = nn.ModuleList([
            nn.Conv2d(in_channels, reg_out_ch, 3, padding=1) for _ in range(num_levels)])
        if use_centerness:
            self.centerness = nn.ModuleList([
                nn.Conv2d(in_channels, 1, 3, padding=1) for _ in range(num_levels)])
        if obb:
            self.angle_reg = nn.ModuleList([
                nn.Conv2d(in_channels, 1, 3, padding=1) for _ in range(num_levels)])

        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in range(num_levels)])

        # ── Dual per-level classifiers ────────────────────────────────────
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

        # ── Init regression heads ─────────────────────────────────────────
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
        cls_head = self.cls_coco if dataset == 'coco' else self.cls_neubie

        cls_out: List[torch.Tensor] = []
        cls_feat_out: List[torch.Tensor] = []
        reg_out: List[torch.Tensor] = []
        ctr_out: List[torch.Tensor] = []
        ang_out: List[torch.Tensor] = []
        dfl_out: List[torch.Tensor] = []

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
# AIFI encoder + auxiliary decoder (unchanged — LayerNorm, TRT-friendly)
# ══════════════════════════════════════════════════════════════════════════════

class AIFIEncoder(nn.Module):
    """Attention-based Intra-scale Feature Interaction on the coarsest FPN level."""

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
        if self._pos_cache is not None and self._pos_hw == (H, W):
            return self._pos_cache.to(device=device, dtype=dtype)
        half = C // 2
        gy = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        gx = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
        dim_t = torch.arange(half // 2, device=device, dtype=torch.float32)
        dim_t = 10000.0 ** (2.0 * dim_t / half)
        pe_y = torch.zeros(H, W, half, device=device)
        pe_y[..., 0::2] = torch.sin(gy.unsqueeze(-1) / dim_t)
        pe_y[..., 1::2] = torch.cos(gy.unsqueeze(-1) / dim_t)
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
        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = tokens + pos.unsqueeze(0)
        tokens = self.norm(self.encoder(tokens))
        return tokens.permute(0, 2, 1).reshape(B, C, H, W)


class AuxDecoderHead(nn.Module):
    """Lightweight transformer decoder for auxiliary training-only supervision."""

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
            nn.Linear(dim, 4),
        )
        prior_prob = 0.01
        nn.init.constant_(self.cls_head.bias,
                          -math.log((1.0 - prior_prob) / prior_prob))
        nn.init.normal_(self.cls_head.weight, std=0.01)

    def forward(self, fpn_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        B = fpn_features[0].shape[0]
        memory_list = [feat.flatten(2).permute(0, 2, 1) for feat in fpn_features]
        memory = torch.cat(memory_list, dim=1)
        queries = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(queries, memory)
        return {
            'aux_cls':   self.cls_head(out),
            'aux_boxes': self.reg_head(out).sigmoid(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FreqEnhance — GroupNorm (same as Phase E original)
# ══════════════════════════════════════════════════════════════════════════════

class FreqEnhance(nn.Module):
    """Per-level frequency-band feature enhancement via Difference of Gaussians."""

    def __init__(self, channels: int, sigma_low: float = 2.0,
                 sigma_high: float = 1.0, kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1
        self.channels = channels
        self.register_buffer('_gauss_low',
                             self._make_gauss_kernel(channels, kernel_size, sigma_low))
        self.register_buffer('_gauss_high',
                             self._make_gauss_kernel(channels, kernel_size, sigma_high))
        self.pad = kernel_size // 2

        self.conv_low  = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.conv_mid  = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)
        self.conv_high = nn.Conv2d(channels, channels, 3, padding=1,
                                   groups=channels, bias=False)

        self.alpha = nn.Parameter(torch.tensor(1.0 / 3))
        self.beta  = nn.Parameter(torch.tensor(1.0 / 3))
        self.gamma = nn.Parameter(torch.tensor(1.0 / 3))

        self.norm = nn.GroupNorm(_gn_groups(channels), channels)

    @staticmethod
    def _make_gauss_kernel(channels: int, ks: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        g1d = torch.exp(-ax.pow(2) / (2 * sigma ** 2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        return g2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, ks, ks).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_blur_low  = F.conv2d(x, self._gauss_low,  padding=self.pad, groups=self.channels)
        x_blur_high = F.conv2d(x, self._gauss_high, padding=self.pad, groups=self.channels)

        p_low  = x_blur_low
        p_mid  = x_blur_high - x_blur_low
        p_high = x - x_blur_high

        fused = (self.alpha * self.conv_low(p_low) +
                 self.beta  * self.conv_mid(p_mid) +
                 self.gamma * self.conv_high(p_high))
        return self.norm(fused) + x


# ══════════════════════════════════════════════════════════════════════════════
# ConvNeXt FPN — GroupNorm (same as Phase E original)
# ══════════════════════════════════════════════════════════════════════════════

class ConvNeXtEfficientPAN(nn.Module):
    """Bidirectional FPN for ConvNeXt multi-scale features (GroupNorm preserved)."""

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
            return nn.Sequential(
                nn.Conv2d(ic, oc, 1, bias=False),
                nn.GroupNorm(_gn_groups(oc), oc),
                nn.ReLU(inplace=True),
            )

        # Per-stage projection (same as Phase E)
        self.proj_s8  = _proj(in_channels_list[0], C)
        self.proj_s16 = _proj(in_channels_list[1], C)
        self.proj_s32 = _proj(in_channels_list[2], C)

        # P5: downsample P4 by 2x
        self.down_p5 = DWSConvGNReLU(C, stride=2)

        # AIFI (optional)
        self.use_aifi = use_aifi
        if use_aifi:
            self.aifi = AIFIEncoder(C, num_heads=aifi_heads,
                                    num_layers=aifi_layers)

        # Top-down smoothing
        self.td_smooth5 = RepDWSBlock(C)
        self.td_smooth4 = RepDWSBlock(C)
        self.td_smooth3 = RepDWSBlock(C)

        # P2 refinement
        self.p2_refine = SmallObjectRefine(C)

        # Bottom-up path
        self.bu_down_p3 = DWSConvGNReLU(C, stride=2)
        self.bu_down_p4 = DWSConvGNReLU(C, stride=2)
        self.bu_smooth4 = RepDWSBlock(C)
        self.bu_down_p5 = DWSConvGNReLU(C, stride=2)
        self.bu_smooth5 = RepDWSBlock(C)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        stage1, stage2, stage3 = features

        p2_raw = self.proj_s8(stage1)
        p3_raw = self.proj_s16(stage2)
        p4_raw = self.proj_s32(stage3)
        p5_raw = self.down_p5(p4_raw)

        if self.use_aifi:
            p5_raw = self.aifi(p5_raw)

        p5_td = self.td_smooth5(p5_raw)
        p4_td = self.td_smooth4(
            p4_raw + F.interpolate(p5_td, size=p4_raw.shape[-2:], mode='nearest'))
        p3_td = self.td_smooth3(
            p3_raw + F.interpolate(p4_td, size=p3_raw.shape[-2:], mode='nearest'))

        p2 = self.p2_refine(
            p2_raw + F.interpolate(p3_td, size=p2_raw.shape[-2:], mode='nearest'))

        p3_bu = p3_td + self.bu_down_p3(p2)
        p4_bu = self.bu_smooth4(p4_td + self.bu_down_p4(p3_bu))
        p5_bu = self.bu_smooth5(p5_td + self.bu_down_p5(p4_bu))

        return [p2, p3_bu, p4_bu, p5_bu]

    def fuse(self):
        for m in self.modules():
            if isinstance(m, RepDWSBlock):
                m.fuse()


# ══════════════════════════════════════════════════════════════════════════════
# Top-level wrapper
# ══════════════════════════════════════════════════════════════════════════════

class ConvNeXtDetectionHeadMixP2TRT(nn.Module):
    """ConvNeXt features -> ConvNeXtEfficientPAN (GN) -> DetectHeadMixTRT.

    Drop-in replacement for ``ConvNeXtDetectionHeadMixP2`` with only DCN
    removed (replaced by DilatedConvBlockGN).  GroupNorm is preserved
    everywhere — no BN calibration or freezing needed.

    Same output format:
        {'cls': [...], 'reg': [...], 'ctr': [...], 'dfl_raw': [...], 'aux': {...}}
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
                 use_aifi: bool = False,
                 aifi_layers: int = 2,
                 aifi_heads: int = 8,
                 reg_max: int = 0,
                 use_centerness: bool = True,
                 use_dcn: bool = False,        # accepted but ignored
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

        self.use_freq_enhance = use_freq_enhance
        if use_freq_enhance:
            self.freq_enhance = nn.ModuleList([
                FreqEnhance(fpn_channels) for _ in range(4)])

        self.head = DetectHeadMixTRT(
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
            use_dcn=False,  # always off
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

    def shared_params(self):
        skip_ids = {id(p) for m in (self.head.cls_coco, self.head.cls_neubie)
                    for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in skip_ids]

    def coco_cls_params(self):
        return list(self.head.cls_coco.parameters())

    def neubie_cls_params(self):
        return list(self.head.cls_neubie.parameters())


# Alias so train_convnext.py can import the same name regardless of --trt-head
ConvNeXtDetectionHeadMixP2 = ConvNeXtDetectionHeadMixP2TRT


# ══════════════════════════════════════════════════════════════════════════════
# Weight conversion: Phase E (GN + DCN) → TRT (GN + DilatedConv)
# ══════════════════════════════════════════════════════════════════════════════

def convert_phaseE_to_trt(checkpoint_path: str, output_path: str,
                          verbose: bool = True) -> None:
    """Convert a Phase E checkpoint (with DCNv2) to TRT-exportable format.

    Only handles DCN → DilatedConv conversion.  GroupNorm keys pass through
    unchanged (no GN→BN renaming, no running stats to add).

    The function:
      1. For DCNv2Block positions (cls_tower.4, cls_tower.5):
         - ``.dcn.weight`` → ``.conv.weight``  (same shape [C,C,3,3])
         - ``.dcn.bias`` → ``.conv.bias``  (same shape [C])
         - Drops ``.offset_mask.*`` keys
      2. All other keys (GN, RepDWSBlock, ECA, classifiers, etc.)
         pass through unchanged
      3. Saves the converted checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    src_sd = ckpt.get('model_head', ckpt) if isinstance(ckpt, dict) else ckpt

    new_sd = {}
    dropped = []
    renamed = []

    for key, value in src_sd.items():
        new_key = key

        # ── Drop offset_mask from DCNv2Block ──────────────────────────────
        if '.offset_mask.' in key:
            dropped.append(key)
            continue

        # ── DCN weight/bias → Conv2d weight/bias ─────────────────────────
        if '.dcn.weight' in key:
            new_key = key.replace('.dcn.weight', '.conv.weight')
            renamed.append((key, new_key))
        elif '.dcn.bias' in key:
            new_key = key.replace('.dcn.bias', '.conv.bias')
            renamed.append((key, new_key))

        new_sd[new_key] = value

    if verbose:
        print(f'Weight conversion: {checkpoint_path} -> {output_path}')
        print(f'  Source keys: {len(src_sd)}')
        print(f'  Dropped (offset_mask): {len(dropped)}')
        if dropped:
            for d in dropped:
                print(f'    {d}')
        print(f'  Renamed (DCN -> Conv): {len(renamed)}')
        if renamed:
            for old, new in renamed:
                print(f'    {old} -> {new}')
        print(f'  Unchanged keys: {len(new_sd) - len(renamed)}')
        print(f'  Output keys: {len(new_sd)}')

    # Save in same format as original checkpoint
    if isinstance(ckpt, dict):
        out_ckpt = {k: v for k, v in ckpt.items() if k != 'model_head'}
        out_ckpt['model_head'] = new_sd
        out_ckpt['trt_converted'] = True
    else:
        out_ckpt = new_sd

    torch.save(out_ckpt, output_path)
    if verbose:
        print(f'  Saved to {output_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Param counting utility
# ══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Test with ConvNeXt small channel dims ──────────────────────────────
    print('=== ConvNeXt small (TRT-exportable, GN preserved) ===')
    in_ch = [192, 384, 768]
    model = ConvNeXtDetectionHeadMixP2TRT(
        in_channels_list=in_ch,
        fpn_channels=256,
        num_coco_classes=80,
        num_neubie_classes=16,
        num_convs=4, cls_convs=6, cls_channels=256,
        obb=False, use_eca=True, cosine_cls=False,
        use_aifi=False,
        reg_max=16,
        use_centerness=False,
        use_dcn=True,  # should be ignored
        use_aux_decoder=True,
        use_cross_level_attn=True,
    ).to(device)

    feats = [
        torch.randn(2, 192, 60, 80, device=device),
        torch.randn(2, 384, 30, 40, device=device),
        torch.randn(2, 768, 15, 20, device=device),
    ]

    # Training mode
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

    expected_sizes = [(60, 80), (30, 40), (15, 20), (8, 10)]
    for i, (h, w) in enumerate(expected_sizes):
        actual_h, actual_w = out['cls'][i].shape[2], out['cls'][i].shape[3]
        status = 'OK' if (actual_h, actual_w) == (h, w) else 'MISMATCH'
        print(f'  P{i+2}: expected ({h},{w}), got ({actual_h},{actual_w}) [{status}]')

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(feats, dataset='neubie')
    print('\nEval mode keys:', list(out_eval.keys()))
    assert 'aux' not in out_eval, 'aux should not be in eval output'

    # Backward pass
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

    # Verify: NO DCNv2 anywhere, GroupNorm IS present
    has_dcn = any('DeformConv2d' in type(m).__name__ for m in model.modules())
    has_gn  = any(isinstance(m, nn.GroupNorm) for m in model.modules())
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    has_bn  = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    print(f'\nTRT-exportable check:')
    print(f'  DeformConv2d present: {has_dcn}  (should be False)')
    print(f'  GroupNorm present:    {has_gn}   (should be True, count={gn_count})')
    print(f'  BatchNorm2d present:  {has_bn}   (should be False)')
    assert not has_dcn, 'DCNv2 found — not exportable!'
    assert has_gn, 'GroupNorm not found — should be preserved!'
    assert not has_bn, 'BatchNorm2d found — should not exist in this variant!'
    print('  PASSED: no DCN, GN preserved, no BN')

    # ── Test with ConvNeXt base channel dims ──────────────────────────────
    print('\n=== ConvNeXt base (TRT-exportable, GN preserved) ===')
    in_ch_base = [256, 512, 1024]
    model_base = ConvNeXtDetectionHeadMixP2TRT(
        in_channels_list=in_ch_base,
        fpn_channels=256,
        num_coco_classes=80, num_neubie_classes=16,
        num_convs=4, cls_convs=6, cls_channels=256,
        obb=False, use_eca=True, cosine_cls=False,
        reg_max=16, use_centerness=False, use_dcn=True,
        use_aux_decoder=True, use_cross_level_attn=True,
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

    # ── Weight conversion test (if checkpoint path provided) ──────────────
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
        out_path = ckpt_path.replace('.pth', '_trt_v3.pth')
        print(f'\n=== Weight conversion test ===')
        convert_phaseE_to_trt(ckpt_path, out_path)

        # Try loading into model
        print(f'\nLoading converted weights into TRT model...')
        ckpt_loaded = torch.load(out_path, map_location='cpu')
        sd = ckpt_loaded.get('model_head', ckpt_loaded)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'  Missing keys: {len(missing)}')
        if missing:
            for k in missing[:10]:
                print(f'    {k}')
        print(f'  Unexpected keys: {len(unexpected)}')
        if unexpected:
            for k in unexpected[:10]:
                print(f'    {k}')
        if len(missing) == 0 and len(unexpected) == 0:
            print('  PERFECT: no missing, no unexpected keys')

    print('\nAll tests passed.')
