# config_mix.py — ViT-S+ dual-classifier mix pipeline (3-level, no P2)
#
# This is the config used by the mix_best_20260604 run (best_e182, best_e188, best_e250).
# Reconstructed from results/mixed-dataset-training/mix_best_20260604/run_config.json.
#
# Architecture: DINOv3 ViT-S+/16 (frozen) + 3-level [P3,P4,P5] DetectHeadMix
# Best checkpoint: best_e250.pth (macro-F1=0.6919)
# Key checkpoints: best_e182 (F1=0.6868), best_e188 (F1=0.6859, best SAHI TL recall)

# ── Dataset paths ─────────────────────────────────────────────────────────────
NEUBIE_ROOT   = '/media/data/shared/merged_coco'   # local SSD (updated from old NFS path)
COCO_ROOT_MIX = '/data2/datasets/coco'

IMG_SIZE   = (480, 640)   # (H, W)
PATCH_SIZE = 32
PROB_AUGMENT_TRAINING = 0.5
PROB_AUGMENT_VALID    = 0.0
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ── Augmentation ──────────────────────────────────────────────────────────────
USE_MOSAIC          = True
MOSAIC_PROB         = 0.5
CLOSE_MOSAIC_EPOCHS = 10
HFLIP_PROB          = 0.5
TRANSLATE_PROB      = 0.5
TRANSLATE_MAX       = 0.1
HUE_DELTA           = 10   # original value (NOT reduced — predates hue-aug fix)
SATURATION_RANGE    = (0.8, 1.2)

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

N_LAYERS_UNFREEZE = 0

# ── Head ──────────────────────────────────────────────────────────────────────
FPN_CH  = 192
N_CONVS = 4

# ── Class counts ──────────────────────────────────────────────────────────────
NUM_COCO_CLASSES   = 80
NUM_NEUBIE_CLASSES = 16

# ── 4-stage curriculum ────────────────────────────────────────────────────────
#   Stage 1: epoch   1– 50  COCO warmup
#   Stage 2: epoch  51– 70  Neubie cls warmup
#   Stage 3: epoch  71–230  Mixed COCO+Neubie (shifting ratio)
#   Stage 4: epoch 231–250  Neubie-only calibration
STAGE1_END = 50
STAGE2_END = 70
STAGE3_END = 230
NUM_EPOCHS = 250

STAGE3_SUB1_END = 110
STAGE3_SUB2_END = 145

# ── Learning rates ────────────────────────────────────────────────────────────
LR_SHARED        = 2e-4
LR_SHARED_STAGE2 = 2e-5
LR_CLS_COCO      = 1e-3
LR_CLS_NEUBIE    = 1e-3

WEIGHT_DECAY     = 0.0001
CLS_WEIGHT_DECAY = 1e-5
LR_WARMUP_EPOCHS = 5
LR_MIN           = 1e-6
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
VFL_CW_NEG = False

USE_OBB = False

# TAL assigner
TAL_TOPK         = 12
TAL_ALPHA        = 0.5
TAL_BETA         = 4.0
PROG_LOSS_EPOCHS = 10

# ── EMA ───────────────────────────────────────────────────────────────────────
USE_EMA   = True
EMA_DECAY = 0.999
EMA_TAU   = 2000

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
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
VAL_METRIC_EVERY = 3

BEST_METRIC = 'f1'
