# config_convnext_small.py — ConvNeXt Small backbone + mix_p2 strategy
#
# Copied from config_mix_p2.py with ConvNeXt-specific changes:
#   - BACKBONE_TYPE = 'convnext'
#   - DINO_MODEL = 'dinov3_convnext_small'
#   - CONVNEXT_IN_CHANNELS = [192, 384, 768]  (stages 1, 2, 3)
#   - WARM_START_CKPT = ''  (training from scratch with new backbone)
#   - USE_AIFI = False  (per plan: Phase C minus AIFI)
#   - BATCH_SIZE = 64

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
MOSAIC_PROB         = 0.3
CLOSE_MOSAIC_EPOCHS = 10
HFLIP_PROB          = 0.5
TRANSLATE_PROB      = 0.5
TRANSLATE_MAX       = 0.1
HUE_DELTA           = 2
SATURATION_RANGE    = (0.9, 1.1)

# ── Small-object augmentations ────────────────────────────────────────────────
ZOOMOUT_PROB        = 0.0
ZOOMOUT_MAX_SCALE   = 2.5
COPYPASTE_PROB         = 0.5
COPYPASTE_MAX_PER_IMG  = 5
COPYPASTE_CLASSES      = ('bollard', 'scooter', 'warning_light',
                          'traffic_light_red', 'traffic_light_green',
                          'traffic_light_other')
COPYPASTE_SCALE_RANGE  = (0.6, 1.4)
SAHICROP_PROB       = 0.25
SAHICROP_UPPER_FRAC = 0.55
SAHICROP_COLS       = 2
SAHICROP_OVERLAP    = 0.2
SAHICROP_MIN_VIS    = 0.3

# ── Backbone ──────────────────────────────────────────────────────────────────
DINOV3_DIR   = '/home/raza/Raza/neubi/DINOv3-object-detection/dinov3'
DINO_MODEL   = 'dinov3_convnext_small'
DINO_WEIGHTS = '/home/raza/Raza/neubi/DINOv3-object-detection/checkpoints/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth'

# ConvNeXt-specific: backbone type and per-stage channel counts
BACKBONE_TYPE = 'convnext'
CONVNEXT_IN_CHANNELS = [192, 384, 768]   # stages 1, 2, 3

# These are kept for API compatibility with train scripts that reference them.
# ConvNeXt doesn't use n_layers/embed_dim the same way ViT does, but the
# training script reads them for logging.
MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16':     12, 'dinov3_vits16plus': 12, 'dinov3_vitb16': 12,
    'dinov3_vitl16':     24, 'dinov3_vith16plus': 32, 'dinov3_vit7b16': 40,
    'dinov3_convnext_tiny':  4, 'dinov3_convnext_small': 4,
    'dinov3_convnext_base':  4, 'dinov3_convnext_large': 4,
}
MODEL_TO_EMBED_DIM = {
    'dinov3_vits16':     384, 'dinov3_vits16plus': 384, 'dinov3_vitb16': 768,
    'dinov3_vitl16':     1024, 'dinov3_vith16plus': 1280, 'dinov3_vit7b16': 4096,
    'dinov3_convnext_tiny':  768, 'dinov3_convnext_small': 768,
    'dinov3_convnext_base':  1024, 'dinov3_convnext_large': 1536,
}

N_LAYERS_UNFREEZE = 0

# ── Head ──────────────────────────────────────────────────────────────────────
FPN_CH  = 256
N_CONVS = 4

# ── P2 (stride-8) + unfreeze knobs ──────────────────────────────────────────
USE_P2          = True
UNFREEZE_LAST_N = 0
LR_BACKBONE     = 1e-5
BACKBONE_WD     = 1e-4
# No warm-start — training from scratch with ConvNeXt backbone
WARM_START_CKPT = ''

# ── Class counts (FIXED) ──────────────────────────────────────────────────────
NUM_COCO_CLASSES   = 80
NUM_NEUBIE_CLASSES = 16

# ── FROM-SCRATCH 4-stage schedule (compressed to 180 epochs) ────────────────
STAGE1_END = 36
STAGE2_END = 50
STAGE3_END = 166
NUM_EPOCHS = 180

# Stage 3 mixing-ratio sub-stages (proportionally scaled)
STAGE3_SUB1_END = 79
STAGE3_SUB2_END = 104

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
VFL_Q_FLOOR = 0.3

USE_OBB = False

# TAL assigner
TAL_TOPK         = 16
TAL_ALPHA        = 0.5
TAL_BETA         = 4.0
PROG_LOSS_EPOCHS = 10

# ── Phase C: Architectural improvements (minus AIFI) ─────────────────────────
USE_AIFI          = False     # DISABLED per plan
AIFI_LAYERS       = 2
AIFI_HEADS        = 8

USE_DFL           = True
DFL_REG_MAX       = 16

USE_CENTERNESS    = False

USE_DCN           = True

USE_AUX_DECODER   = True
AUX_NUM_QUERIES   = 100
AUX_DECODER_LAYERS = 2
AUX_LOSS_WEIGHT   = 0.5

# ── EMA ───────────────────────────────────────────────────────────────────────
USE_EMA   = True
EMA_DECAY = 0.999
EMA_TAU   = 2000

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 112       # 48GB GPU 1: peaks ~38GB worst-case, ~10GB headroom
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
VAL_METRIC_EVERY = 10

BEST_METRIC = 'f1'
