# model_backbone_v3.py — Multi-layer backbone
#
# v2 DinoBackbone returns only feats[-1]  (the final ViT block).
# v3 DinoBackboneV3 returns THREE feature maps from different depths:
#
#   [shallow, mid, deep]
#   shallow : block  ~n/4   (local textures / edges) → seeds P3 (small objects)
#   mid     : block  ~n/2   (part-level features)    → seeds P4
#   deep    : block  n-1    (global semantics)        → seeds P5 (large objects)
#
# All three are at the same spatial resolution (H/patch × W/patch = H/16 × W/16).
# The MultiLevelEfficientPAN in model_head_v3.py handles the multi-scale FPN build.

import torch
import torch.nn as nn


class DinoBackbone(nn.Module):
    """Original single-output backbone — kept for backward compatibility."""

    def __init__(self, dino_model, n_layers=12):
        super().__init__()
        self.dino     = dino_model
        self.n_layers = n_layers

    def forward(self, x):
        feats = self.dino.get_intermediate_layers(
            x, n=range(self.n_layers), reshape=True, norm=True)
        return feats[-1]


class DinoBackboneV3(nn.Module):
    """Multi-layer backbone: returns [shallow, mid, deep] feature maps.

    All outputs have shape (B, embed_dim, H/patch, W/patch).
    The three indices are:
        i_shallow = n_layers // 4
        i_mid     = n_layers // 2
        i_deep    = n_layers - 1

    For ViT-S/B 12-layer: layers [2, 6, 11]
    For ViT-L 24-layer  : layers [6, 12, 23]
    """

    def __init__(self, dino_model, n_layers=12):
        super().__init__()
        self.dino     = dino_model
        self.n_layers = n_layers

        self.i_shallow = max(0, n_layers // 4)
        self.i_mid     = max(0, n_layers // 2)
        self.i_deep    = n_layers - 1

    def forward(self, x):
        """Returns list [shallow_feat, mid_feat, deep_feat], each (B, C, Hf, Wf)."""
        feats = self.dino.get_intermediate_layers(
            x, n=range(self.n_layers), reshape=True, norm=True)
        return [feats[self.i_shallow], feats[self.i_mid], feats[self.i_deep]]
