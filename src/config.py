"""Central configuration for the WiFi CSI sensing project.

All paths, the CSI signal geometry, the class set and model hyper-parameters
live here so the rest of the codebase stays declarative. The defaults match
what an **ESP32 (HT20, single antenna)** reports via Espressif's `esp-csi`
firmware, so swapping synthetic data for real captures — or a public dataset —
is mostly a path change.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # drop real captures / public datasets here
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SYNTHETIC_NPZ = DATA_DIR / "synthetic_csi.npz"   # generated dataset
REPLAY_CSV = DATA_DIR / "replay_stream.csv"      # held-out stream for the live demo

MODEL_PATH = MODELS_DIR / "best_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
TORCH_MODEL_PATH = MODELS_DIR / "cnn_lstm.pt"
METRICS_PATH = REPORTS_DIR / "metrics.json"

for _d in (DATA_DIR, RAW_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# CSI signal geometry (ESP32, HT20)
# --------------------------------------------------------------------------- #
# The ESP32 reports CSI as int8 I/Q pairs per subcarrier. HT20 → 64 subcarriers
# in raw FFT order; the DC bin and guard bands carry no data and are dropped.
NUM_SUBCARRIERS = 64
# Non-null subcarrier indices (data + pilots) in raw FFT order, 52 total.
VALID_SUBCARRIERS = list(range(1, 27)) + list(range(38, 64))

SAMPLE_RATE_HZ = 100          # packet/CSI rate the TX is driven at
WINDOW_SEC = 2.0              # activity-recognition window
WINDOW_SIZE = int(WINDOW_SEC * SAMPLE_RATE_HZ)        # 200 samples
WINDOW_STEP = WINDOW_SIZE // 4                        # 75% overlap

# Respiration needs a much longer window for usable frequency resolution.
RESP_WINDOW_SEC = 20.0
RESP_WINDOW_SIZE = int(RESP_WINDOW_SEC * SAMPLE_RATE_HZ)
RESP_BAND_HZ = (0.1, 0.7)     # 6–42 breaths/min

# --------------------------------------------------------------------------- #
# Classes  (tiered so one model covers presence → activity → breathing)
# --------------------------------------------------------------------------- #
#   Tier 1 (presence)  : empty  vs  {everything else}
#   Tier 2 (activity)  : sit / stand / walk / fall
#   Tier 3 (breathing) : `breathe` windows feed the respiration-rate estimator
CLASSES = ["empty", "sit", "stand", "walk", "fall", "breathe"]
PRESENCE_POSITIVE = [c for c in CLASSES if c != "empty"]

# Map the granular labels of public HAR datasets onto our coarse class set.
# Used by datasets.normalize_labels() when you load e.g. UT-HAR.
PUBLIC_LABEL_MAP = {
    "noactivity": "empty", "empty": "empty", "nobody": "empty",
    "sitdown": "sit", "sit": "sit",
    "stand": "stand", "standup": "stand", "standing": "stand",
    "walk": "walk", "walking": "walk", "run": "walk", "pickup": "walk",
    "fall": "fall", "falling": "fall", "falldown": "fall",
    "liedown": "breathe", "lie": "breathe", "breathe": "breathe",
}

# --------------------------------------------------------------------------- #
# Training config
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.25

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=80,
    early_stopping=True,
    random_state=RANDOM_STATE,
)

# Optional 1D-CNN-LSTM (used only if PyTorch is installed).
CNN_PARAMS = dict(epochs=30, batch_size=64, lr=1e-3, weight_decay=5e-4)

# --------------------------------------------------------------------------- #
# Hardware capture (esp-csi)
# --------------------------------------------------------------------------- #
SERIAL_PORT = "/dev/tty.usbserial-0001"   # override on the CLI
SERIAL_BAUD = 921600
UDP_HOST = "0.0.0.0"
UDP_PORT = 5566
