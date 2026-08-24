# model_backbone_convnext.py — ConvNeXt backbone wrapper for DINOv3
#
# Unlike ViT backbones where all intermediate layers share the same resolution
# (H/16, W/16) and channel count (embed_dim), ConvNeXt produces naturally
# multi-scale features with DIFFERENT resolutions AND channel dimensions:
#
#   Stage 0: stride  4, dims[0] channels  (120x160 for 480x640 input)  — skipped
#   Stage 1: stride  8, dims[1] channels  ( 60x80)
#   Stage 2: stride 16, dims[2] channels  ( 30x40)
#   Stage 3: stride 32, dims[3] channels  ( 15x20)
#
# Stage 0 is skipped (stride 4 is too fine-grained for detection, wastes memory).
# The detection head's FPN uses per-stage projection layers to handle the
# different channel counts (unlike ViT where a single in_channels suffices).
#
# ConvNeXt sizes (DINOv3 pretrained):
#   tiny:  dims=[96, 192, 384, 768]    depths=[3, 3, 9, 3]
#   small: dims=[96, 192, 384, 768]    depths=[3, 3, 27, 3]
#   base:  dims=[128, 256, 512, 1024]  depths=[3, 3, 27, 3]
#   large: dims=[192, 384, 768, 1536]  depths=[3, 3, 27, 3]

import torch
import torch.nn as nn


class DinoBackboneConvNeXt(nn.Module):
    """Frozen ConvNeXt backbone returning 3 multi-scale feature maps.

    Returns [stage1, stage2, stage3]:
      stage1: (B, dims[1], H/8,  W/8)   -- stride 8
      stage2: (B, dims[2], H/16, W/16)  -- stride 16
      stage3: (B, dims[3], H/32, W/32)  -- stride 32

    Note: channel counts differ per stage (unlike ViT where all are embed_dim).
    The detection head must use per-stage projection layers.
    """

    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model

    def forward(self, x):
        """Returns list [stage1_feat, stage2_feat, stage3_feat].

        Each tensor has shape (B, C_stage, H_stage, W_stage) where C_stage
        and spatial dims vary per stage.
        """
        # n=range(4) requests all 4 stages; reshape=True keeps (B, C, H, W)
        feats = self.dino.get_intermediate_layers(
            x, n=range(4), reshape=True, norm=True)
        # Skip stage 0 (stride 4) — too fine for detection
        return [feats[1], feats[2], feats[3]]
