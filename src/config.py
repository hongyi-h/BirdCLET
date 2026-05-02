DATA_DIR = "data"
TRAIN_AUDIO_DIR = "data/train_audio"
TRAIN_SOUNDSCAPES_DIR = "data/train_soundscapes"
TEST_SOUNDSCAPES_DIR = "data/test_soundscapes"
PRECOMPUTED_DIR = "data/precomputed"

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
