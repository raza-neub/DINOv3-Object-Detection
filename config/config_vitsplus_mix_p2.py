# config_mix_p2.py — Phase 2 (P2 stride-8) + optional Phase 3 (unfreeze) config
#
# Standalone copy of config_mix.py with the changes documented in
# docs/unfreeze_backbone_edgecrafter_design.md §14. Used by train/train_mix_p2.py.
#
# Key differences vs config_mix.py:
#   - WARM-START from best_e182 (not a fresh 250-epoch run)
#   - Collapsed/shortened schedule: mixed Stage-3 block + short Stage-4 calibration
#   - BATCH_SIZE 64 (STAL-anchor 4× memory safeguard — see §14.6)
#   - P2 + unfreeze knobs (USE_P2, UNFREEZE_LAST_N, LR_BACKBONE, BACKBONE_WD)

# ── Dataset paths ─────────────────────────────────────────────────────────────
NEUBIE_ROOT   = '/media/data/shared/merged_coco'
COCO_ROOT_MIX = '/data2/datasets/coco'

IMG_SIZE   = (480, 640)   # (H, W)
PATCH_SIZE = 32
PROB_AUGMENT_TRAINING = 0.5
PROB_AUGMENT_VALID    = 0.0
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Augmentation ──────────────────────────────────────────────────────────────
USE_MOSAIC          = True
MOSAIC_PROB         = 0.3      # reduced — mosaic halves object sizes, hurts tiny objects
CLOSE_MOSAIC_EPOCHS = 10
HFLIP_PROB          = 0.5
TRANSLATE_PROB      = 0.5
TRANSLATE_MAX       = 0.1
# Hue jitter kept low — the label IS the hue for traffic_light_* (OpenCV H 0-179).
HUE_DELTA           = 2
SATURATION_RANGE    = (0.9, 1.1)

# ── Small-object augmentations (NEW — the small-class lever) ────────────────────
# RandomZoomOut: shrink the frame into a mean-padded canvas → more small instances.
ZOOMOUT_PROB        = 0.0   # DROPPED: zoom-out shrinks already-tiny traffic lights
                            # (8px → ~3px), training them as unlearnable background —
                            # a likely contributor to the v199-run small-class collapse.
ZOOMOUT_MAX_SCALE   = 2.5
# Copy-paste: paste extra instances of the crucial small/rare classes. Neubie-only
# (COCO has none of these names → silent no-op there). Names must match annotations.
COPYPASTE_PROB         = 0.5   # from-scratch run — no warm-init concern
COPYPASTE_MAX_PER_IMG  = 5
COPYPASTE_CLASSES      = ('bollard', 'scooter', 'warning_light',
                          'traffic_light_red', 'traffic_light_green',
                          'traffic_light_other')
COPYPASTE_SCALE_RANGE  = (0.6, 1.4)
# SAHI-crop: upper-strip tile up-sampled to IMG_SIZE — mirrors inference SAHI
# geometry so training matches the magnified distribution SAHI feeds at inference.
SAHICROP_PROB       = 0.25
SAHICROP_UPPER_FRAC = 0.55     # mirrors inference --sahi-upper-frac
SAHICROP_COLS       = 2        # mirrors inference --sahi-cols
SAHICROP_OVERLAP    = 0.2      # mirrors inference --sahi-overlap
SAHICROP_MIN_VIS    = 0.3      # drop a box if a crop leaves <30% of its area

# ── Backbone ──────────────────────────────────────────────────────────────────
DINOV3_DIR   = '/home/raza/Raza/neubi/DINOv3-object-detection/dinov3'
DINO_MODEL   = 'dinov3_vits16plus'
DINO_WEIGHTS = '/data2/Omer/DinoV3_checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'

MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16':     12, 'dinov3_vits16plus': 12, 'dinov3_vitb16': 12,
    'dinov3_vitl16':     24, 'dinov3_vith16plus': 32, 'dinov3_vit7b16': 40,
}
MODEL_TO_EMBED_DIM = {
    'dinov3_vits16':     384, 'dinov3_vits16plus': 384, 'dinov3_vitb16': 768,
    'dinov3_vitl16':     1024, 'dinov3_vith16plus': 1280, 'dinov3_vit7b16': 4096,
}

N_LAYERS_UNFREEZE = 0   # legacy knob; the P2 script uses UNFREEZE_LAST_N below

# ── Head ──────────────────────────────────────────────────────────────────────
FPN_CH  = 256
N_CONVS = 4

# ── P2 (stride-8) + unfreeze knobs (NEW) ──────────────────────────────────────
USE_P2          = True     # add the stride-8 P2 level (4-level pyramid)
# Phase 2 = 0 (backbone frozen); Phase 3 = 2 (unfreeze last 2 blocks).
# CLI --unfreeze-last-n overrides this.
UNFREEZE_LAST_N = 0
LR_BACKBONE     = 1e-5     # ≈ LR_SHARED / 20 (EdgeCrafter anchor 2.5e-5)
BACKBONE_WD     = 1e-4     # weights only; norm/bias get 0
# Warm-start note:
# best_e250 was trained with ViT-S+ (embed=384, fpn_ch=192, 3-level P3/P4/P5).
# Phase C uses fpn_ch=256 + P2 (4-level) + DFL + DCNv2 + AIFI + aux_decoder,
# so FPN projectors and new modules have shape mismatches → fresh init.
# The cls_tower and cosine classifiers (cls_channels=256) match → warm-start.
# Set WARM_START_CKPT='' to train fully from scratch (full 4-stage schedule then).
# Phase C: warm-start from ViT-S+ best_e250 (3-level, fpn_ch=192).
# Shape-filtered loading: cls_tower + cosine classifiers load (matching shapes);
# FPN projectors (384→256 vs 384→192), bbox_reg (DFL), DCNv2 blocks, AIFI,
# aux_decoder, P2-level modules all start fresh (shape mismatch → re-init).
WARM_START_CKPT = '/home/raza/Raza/neubi/DINOv3-object-detection/results/mixed-dataset-training/mix_best_20260604/best_e250.pth'

# ── Class counts (FIXED) ──────────────────────────────────────────────────────
NUM_COCO_CLASSES   = 80
NUM_NEUBIE_CLASSES = 16

# ── FROM-SCRATCH full 4-stage schedule ───────────────────────────────────────
#   Stage 1: epoch   1– 50  COCO warmup
#   Stage 2: epoch  51– 70  Neubie cls warmup
#   Stage 3: epoch  71–230  Mixed COCO+Neubie (shifting ratio)
#   Stage 4: epoch 231–250  Neubie-only calibration
STAGE1_END = 50
STAGE2_END = 70
STAGE3_END = 230
NUM_EPOCHS = 250

# Stage 3 mixing-ratio sub-stages (0-indexed epoch boundaries):
#   70–109 : 60% COCO, 40% Neubie
#  110–144 : 40% COCO, 60% Neubie
#  145–229 : 20% COCO, 80% Neubie
STAGE3_SUB1_END = 110
STAGE3_SUB2_END = 145

# ── Learning rates ────────────────────────────────────────────────────────────
LR_SHARED        = 2e-4
LR_SHARED_STAGE2 = 2e-5    # unused (no stage 2 here) but kept for API parity
LR_CLS_COCO      = 1e-3
LR_CLS_NEUBIE    = 1e-3

WEIGHT_DECAY     = 0.0001
CLS_WEIGHT_DECAY = 1e-5
LR_WARMUP_EPOCHS = 5        # full warmup — from-scratch random init
LR_MIN           = 1e-6
# Phase 2 (frozen backbone): 5.0 is fine. Phase 3 (unfreeze): train script
# tightens this to 0.1 automatically when UNFREEZE_LAST_N > 0.
GRAD_CLIP_NORM   = 5.0

# ── Loss ──────────────────────────────────────────────────────────────────────
FOCAL_ALPHA  = 0.25
FOCAL_GAMMA  = 2.0
WEIGHT_REG   = 1.0
WEIGHT_CTR   = 1.0
ANGLE_WEIGHT = 0.2

USE_VFL    = True
VFL_ALPHA  = 0.75
VFL_GAMMA  = 2.0
VFL_CW_NEG = False     # weight positives only — preserve rare-class recall
VFL_Q_FLOOR = 0.3      # min positive weight — ensures rare classes get training signal

USE_OBB = False        # axis-aligned — matches best_e182 (no angle branch trained)

# TAL assigner
TAL_TOPK         = 16      # more anchors per GT with 6370 total P2-level anchors
TAL_ALPHA        = 0.5
TAL_BETA         = 4.0
PROG_LOSS_EPOCHS = 10  # full reg/ctr warmup ramp — from-scratch

# ── Phase C: Architectural improvements ─────────────────────────────────────
USE_AIFI          = True      # C1: MHSA encoder on P5 (highest impact)
AIFI_LAYERS       = 2
AIFI_HEADS        = 8

USE_DFL           = True      # C2: Distribution Focal Loss regression
DFL_REG_MAX       = 16        # 16 discrete bins per box side

USE_CENTERNESS    = False     # C3: disable centerness (VFL already IoU-aware)

USE_DCN           = True      # C4: DCNv2 in last 2 cls tower blocks

USE_AUX_DECODER   = True      # C5: auxiliary decoder supervision (training only)
AUX_NUM_QUERIES   = 100
AUX_DECODER_LAYERS = 2
AUX_LOSS_WEIGHT   = 0.5

# ── EMA ───────────────────────────────────────────────────────────────────────
USE_EMA   = True
EMA_DECAY = 0.999
EMA_TAU   = 2000

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64        # ViT-S+: smaller backbone allows larger batch
USE_AMP    = True

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_MODEL   = True
RESULTS_PATH = '/home/raza/Raza/neubi/DINOv3-object-detection/results/mixed-dataset-training'

# ── Inference ─────────────────────────────────────────────────────────────────
MODEL_PATH_INFERENCE = ''
SCORE_THRESH         = 0.2
NMS_THRESH           = 0.6
CLASS_NAMES_PATH     = '/home/raza/Raza/neubi/DINOv3-object-detection/src/class_names.txt'
TOPK_INFER           = 300

# ── Validation ────────────────────────────────────────────────────────────────
VAL_IOU_THRESH   = 0.5
VAL_SCORE_THRESH = 0.2
VAL_NMS_THRESH   = 0.6
VAL_RARE_THRESH  = 0.08
NUM_SAMPLES_PLOT = 6
VAL_METRIC_EVERY = 2     # metrics every other epoch (short run → want them often)

BEST_METRIC = 'f1'       # macro-F1 selection (deployment-aligned)
