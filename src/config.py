import os


DATA_ROOT = os.environ.get("BIRDCLEF_DATA_ROOT", "data")
DATA_DIR = os.environ.get(
    "BIRDCLEF_COMPETITION_DATA_DIR",
    os.path.join(DATA_ROOT, "BirdCLEF+ 2026"),
)
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
TRAIN_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "train_soundscapes")
TEST_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "test_soundscapes")

EXTERNAL_DATA_DIR = DATA_ROOT
PRECOMPUTED_DIR = os.environ.get(
    "BIRDCLEF_PRECOMPUTED_DIR",
    os.path.join(DATA_ROOT, "precomputed"),
)
PSEUDO_LABEL_DIR = os.environ.get("BIRDCLEF_PSEUDO_LABEL_DIR", DATA_ROOT)
PSEUDO_LABEL_PATH = os.path.join(PSEUDO_LABEL_DIR, "pseudo_labels.csv")

PERCH_ONNX_DIR = os.path.join(EXTERNAL_DATA_DIR, "perch-onnx-for-birdclef-2026")
PERCH_ONNX_PATH = os.path.join(PERCH_ONNX_DIR, "perch_v2.onnx")
PERCH_LABELS_PATH = os.path.join(PERCH_ONNX_DIR, "labels.csv")
PERCH_META_DIR = os.path.join(EXTERNAL_DATA_DIR, "perch-meta")
SED_MODEL_DIR = os.path.join(EXTERNAL_DATA_DIR, "bc2026-distilled-sed-public")
TRAIN_AUDIO_HEAD_DIR = os.path.join(EXTERNAL_DATA_DIR, "bird26-train-audio-head-v1")
TRAIN_AUDIO_HEAD_PATH = os.path.join(TRAIN_AUDIO_HEAD_DIR, "head_weights_train_audio.npz")

PRETRAINED_MODEL_DIR = os.environ.get("BIRDCLEF_PRETRAINED_MODEL_DIR", "pretrained_models")
PERCH_TF_MODEL_DIR = os.path.join(PRETRAINED_MODEL_DIR, "bird-vocalization-classifier")
PERCH_TF_LABELS_PATH = os.path.join(PERCH_TF_MODEL_DIR, "assets", "labels.csv")

SR = 32000
DURATION = 5
N_FFT = 1024
HOP_LENGTH = 320
N_MELS = 128
FMIN = 20
FMAX = 16000

NUM_CLASSES = 234
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
SEED = 42

MODEL_NAME = "tf_efficientnet_b0.ns_jft_in1k"
CHECKPOINT_DIR = "checkpoints"
ONNX_PATH = "checkpoints/model.onnx"

BACKBONE_REGISTRY = {
    "b0": "tf_efficientnet_b0.ns_jft_in1k",
    "v2s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "nfnet": "eca_nfnet_l0",
    "regnety": "regnety_016",
}

NON_BIRD_CLASSES = {"Amphibia", "Insecta", "Mammalia", "Reptilia"}
