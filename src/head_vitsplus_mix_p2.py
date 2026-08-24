"""
model_head_mix_p2.py — P2 (stride-8) augmented dual-classifier head
====================================================================

Phase-2 variant of `model_head_mix.py` (the current 256-ch cosine-classifier
head used by the `mix_best_20260604` run). It adds a **4th, finer pyramid level
P2 at stride 8 (60×80)** for small objects (traffic lights, bollard, scooter),
while reusing the existing 256-ch decoupled cls tower, cosine classifiers, and
lightweight reg tower **unchanged** (so it warm-starts cleanly from `best_e182`).

Pyramid (finest-first):  [P2, P3, P4, P5]  →  strides [8, 16, 32, 64]

Design choices that keep loss/assigner/decode/optimizer edit-free:
  - Finest-first ordering → `compute_strides` gives [8,16,32,64], STAL
    `min_stride = strides[0] = 8` is correct.
  - The P2 path adds ONE new neck module (`p2_refine`); the upsample reuses the
    existing `proj_shallow` output, so warm-start has almost nothing random.
  - The head is the *same* `DetectHeadMix`, just `num_levels=4`; its per-level
    ModuleLists gain a 4th entry. `shared_params/coco_cls_params/neubie_cls_params`
    partition by module, so P2 params are picked up automatically.

Warm-start from a 3-level `best_e182.pth`:
  - per-level head modules: old level i (P3,P4,P5) → new level i+1; and the new
    level 0 (P2) is **cloned from old level 0 (P3)** — both are fine levels, so
    this is a far better init than random.
  - shared towers + base-FPN keys load directly; the new `fpn.p2_refine.*` stays
    at init. Use `warm_start_from_3level()` below.

This module imports the building blocks from the existing files (no duplication)
and does NOT modify any existing file.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neck import MultiLevelEfficientPAN, count_parameters
from src.blocks import SmallObjectRefine, DWSConvGNReLU
from src.head_vitsplus_mix import DetectHeadMix   # reuse the current 256-ch cosine head


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
        self._pos_hw: Tuple[int, int] = (0, 0)

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


# ──────────────────────────────────────────────────────────────────────────────
# Neck: MultiLevelEfficientPAN + a stride-8 P2 level
# ──────────────────────────────────────────────────────────────────────────────

class MultiLevelEfficientPANP2(MultiLevelEfficientPAN):
    """Base bidirectional PAN + a synthesized stride-8 P2 level.

    Reuses every base module verbatim (so they load from a 3-level checkpoint).
    Adds exactly one new module: `p2_refine` (2×DWS + ECA residual).

    P2 build:
        p3_raw  = proj_shallow(shallow)                 # stride 16, 30×40  (reused)
        p2_up   = upsample(p3_raw, ×2)                  # stride 8,  60×80
        p2_fuse = p2_up + upsample(p3_td, ×2)           # inject top-down P3 semantics
        P2      = p2_refine(p2_fuse)                     # small-object refine

    Returns [P2, P3_td, P4_bu, P5_bu]  (finest-first).
    """

    def __init__(self, in_channels: int, out_channels: int = 192,
                 use_aifi: bool = False, aifi_layers: int = 2, aifi_heads: int = 8):
        super().__init__(in_channels, out_channels=out_channels)
        self.p2_refine  = SmallObjectRefine(out_channels)
        self.bu_down_p3 = DWSConvGNReLU(out_channels, stride=2)  # P2→P3 bottom-up
        self.use_aifi = use_aifi
        if use_aifi:
            self.aifi = AIFIEncoder(out_channels, num_heads=aifi_heads,
                                    num_layers=aifi_layers)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        shallow, mid, deep = features

        # ----- identical to base MultiLevelEfficientPAN -----
        p3_raw = self.proj_shallow(shallow)
        p4_raw = self.down_mid(self.proj_mid(mid))
        p5_raw = self.down_deep_2(self.down_deep_1(self.proj_deep(deep)))

        # ----- AIFI: self-attention on coarsest level before top-down -----
        if self.use_aifi:
            p5_raw = self.aifi(p5_raw)

        p5_td = self.td_smooth5(p5_raw)
        p4_td = self.td_smooth4(
            p4_raw + F.interpolate(p5_td, size=p4_raw.shape[-2:], mode='nearest'))
        p3_td = self.td_smooth3(
            p3_raw + F.interpolate(p4_td, size=p3_raw.shape[-2:], mode='nearest'))
        p3_td = self.p3_refine(p3_td)

        # ----- new stride-8 P2 (before bottom-up, so P2 feeds back) -----
        p2_h = p3_raw.shape[-2] * 2
        p2_w = p3_raw.shape[-1] * 2
        p2_up   = F.interpolate(p3_raw, size=(p2_h, p2_w), mode='nearest')
        p2_sem  = F.interpolate(p3_td,  size=(p2_h, p2_w), mode='nearest')
        p2      = self.p2_refine(p2_up + p2_sem)

        # ----- bottom-up: P2 → P3 → P4 → P5 -----
        p3_bu = p3_td + self.bu_down_p3(p2)                  # P2 feeds into P3
        p4_bu = self.bu_smooth4(p4_td + self.bu_down_p4(p3_bu))
        p5_bu = self.bu_smooth5(p5_td + self.bu_down_p5(p4_bu))

        return [p2, p3_bu, p4_bu, p5_bu]   # finest-first


# ──────────────────────────────────────────────────────────────────────────────
# Top-level wrapper (4-level)
# ──────────────────────────────────────────────────────────────────────────────

class DINODetectionHeadMixP2(nn.Module):
    """[shallow, mid, deep] → P2-augmented PAN → 4-level DetectHeadMix.

    Same args as the current `DINODetectionHeadMix`; `num_levels` is fixed to 4.
    Phase C additions: AIFI encoder, DFL, centerness removal, DCNv2, aux decoder.
    """

    def __init__(self, backbone_out_channels: int = 384,
                 fpn_channels: int = 192,
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
                 aux_decoder_layers: int = 2):
        super().__init__()
        self.num_levels = 4
        self.fpn  = MultiLevelEfficientPANP2(backbone_out_channels,
                                             out_channels=fpn_channels,
                                             use_aifi=use_aifi,
                                             aifi_layers=aifi_layers,
                                             aifi_heads=aifi_heads)
        self.head = DetectHeadMix(
            in_channels=fpn_channels,
            num_coco_classes=num_coco_classes,
            num_neubie_classes=num_neubie_classes,
            num_convs=num_convs,
            cls_convs=cls_convs,
            cls_channels=cls_channels,
            num_levels=4,                      # ← P2 + P3 + P4 + P5
            obb=obb,
            use_eca=use_eca,
            cosine_cls=cosine_cls,
            cosine_scale=cosine_scale,
            reg_max=reg_max,
            use_centerness=use_centerness,
            use_dcn=use_dcn,
        )
        self.use_aux_decoder = use_aux_decoder
        if use_aux_decoder:
            self.aux_decoder = AuxDecoderHead(
                dim=fpn_channels, num_queries=aux_num_queries,
                num_classes=num_neubie_classes, num_layers=aux_decoder_layers,
            )

    def forward(self, features: List[torch.Tensor],
                dataset: str = 'neubie') -> Dict:
        fpn_out = self.fpn(features)
        head_out = self.head(fpn_out, dataset=dataset)
        if self.training and self.use_aux_decoder:
            head_out['aux'] = self.aux_decoder(fpn_out)
        return head_out

    def fuse(self):
        self.fpn.fuse()
        self.head.fuse()
        return self

    # param-group helpers — identical partitioning to the base head
    def shared_params(self):
        skip_ids = {id(p) for m in (self.head.cls_coco, self.head.cls_neubie)
                    for p in m.parameters()}
        return [p for p in self.parameters() if id(p) not in skip_ids]

    def coco_cls_params(self):
        return list(self.head.cls_coco.parameters())

    def neubie_cls_params(self):
        return list(self.head.cls_neubie.parameters())


# ──────────────────────────────────────────────────────────────────────────────
# Warm-start: load a 3-level best_e182 into this 4-level model
# ──────────────────────────────────────────────────────────────────────────────

# per-level head modules whose level index must shift +1 (and clone P3→P2 at idx 0)
_PER_LEVEL_RE = re.compile(
    r'^(head\.(?:bbox_reg|centerness|angle_reg|cls_coco|cls_neubie|scales))\.(\d+)(.*)$')


def remap_3level_to_4level(sd_3lvl: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Build a 4-level state-dict view from a 3-level checkpoint.

    - per-level head key `....{i}....`  → `....{i+1}....`  (P3,P4,P5 → levels 1,2,3)
    - additionally clone old level 0 (P3) → new level 0 (P2)
    - all other keys (shared towers, base-FPN, cls_entry/eca) copied unchanged
    - new keys with no source (e.g. `fpn.p2_refine.*`) are simply absent → stay at init
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd_3lvl.items():
        m = _PER_LEVEL_RE.match(k)
        if m:
            prefix, idx, suffix = m.group(1), int(m.group(2)), m.group(3)
            out[f'{prefix}.{idx + 1}{suffix}'] = v          # shift +1
            if idx == 0:
                out[f'{prefix}.0{suffix}'] = v.clone()       # P2 ← P3 clone
        else:
            out[k] = v
    return out


def warm_start_from_3level(model: nn.Module,
                           ckpt_path: str,
                           map_location: str = 'cpu') -> Dict[str, int]:
    """Load a checkpoint into a 4-level `DINODetectionHeadMixP2`.

    Auto-detects whether the checkpoint is 3-level (needs remapping) or
    4-level (direct partial load). Handles Phase C architecture changes
    (DFL, DCNv2, AIFI, etc.) by loading only shape-matching keys.

    Returns a report dict with loaded / fresh / mismatch / unexpected counts.
    """
    raw = torch.load(ckpt_path, map_location=map_location)
    sd_ckpt = raw.get('model_head', raw) if isinstance(raw, dict) and 'model_head' in raw else raw

    # Detect 3-level vs 4-level: check if any per-level key has index 3
    is_4level = any(_PER_LEVEL_RE.match(k) and int(_PER_LEVEL_RE.match(k).group(2)) >= 3
                    for k in sd_ckpt.keys())

    if is_4level:
        # Direct load — no index remapping needed
        source = sd_ckpt
    else:
        # 3-level → 4-level: remap indices and clone P3→P2
        source = remap_3level_to_4level(sd_ckpt)

    model_sd = model.state_dict()

    loaded, shape_mismatch = {}, []
    for k, v in source.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loaded[k] = v
        elif k in model_sd:
            shape_mismatch.append(k)
    missing = [k for k in model_sd if k not in loaded]
    unexpected = [k for k in source if k not in model_sd]

    model_sd.update(loaded)
    model.load_state_dict(model_sd, strict=True)

    return {
        'ckpt_keys': len(sd_ckpt),
        'ckpt_type': '4-level' if is_4level else '3-level (remapped)',
        'loaded': len(loaded),
        'fresh_init': len(missing),
        'shape_mismatch': len(shape_mismatch),
        'unexpected': len(unexpected),
        'fresh_keys': missing,
        'unexpected_keys': unexpected,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Self-test: build, forward (4 levels), warm-start from best_e182
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    ckpt = (sys.argv[1] if len(sys.argv) > 1 else
            'results/mixed-dataset-training/mix_best_20260604/best_e182.pth')

    model = DINODetectionHeadMixP2(
        backbone_out_channels=384, fpn_channels=192,
        num_coco_classes=80, num_neubie_classes=16,
        num_convs=4, cls_convs=6, cls_channels=256,
        obb=False, use_eca=True, cosine_cls=True,
    )

    # forward: 480×640 → backbone 30×40 → expect P2 60×80 ... P5 8×10
    feats = [torch.randn(1, 384, 30, 40) for _ in range(3)]
    model.eval()
    with torch.no_grad():
        out = model(feats, dataset='neubie')
    print('levels:', len(out['cls']))
    print('cls shapes:', [tuple(t.shape) for t in out['cls']])
    print('reg shapes:', [tuple(t.shape) for t in out['reg']])

    rep = warm_start_from_3level(model, ckpt)
    print('\nwarm-start report:')
    for kk in ('ckpt_keys', 'loaded', 'fresh_init', 'shape_mismatch', 'unexpected'):
        print(f'  {kk:16s}: {rep[kk]}')
    print('  fresh keys (P2, expected):', rep['fresh_keys'])
    print('  unexpected (should be []):', rep['unexpected_keys'])

    p = count_parameters(model)
    print(f'\nparams total: {p["total"]/1e6:.3f} M')
