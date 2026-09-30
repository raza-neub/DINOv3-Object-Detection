"""
model_head_convnext.py — ConvNeXt-backbone detection head (mix_p2 strategy)
===========================================================================

Standalone detection head for ConvNeXt backbones. Architecturally parallel to
`model_head_mix_p2.py` (which handles ViT backbones) — same output format,
same loss/decode/optimizer compatibility, but a different FPN that exploits
ConvNeXt's naturally multi-scale features.

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

This module does NOT modify any existing file. It imports building blocks
from model_head_v2.py, model_head_v3.py, model_head_mix.py, and
model_head_mix_p2.py.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_head_v2 import (
    ConvGNReLU,
    DWSConvGNReLU,
    RepDWSBlock,
    SmallObjectRefine,
    count_parameters,
)
from src.model_head_mix import DetectHeadMix
from src.model_head_mix_p2 import AIFIEncoder, AuxDecoderHead


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
