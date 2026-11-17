from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

SEED = 42
UNKNOWN_REAL = "Infiltration"
CLASS_CAP = 10000
TRAIN_CAP = 10000
WINDOW_SIZE = 500
RETRAIN_DELAY = 1
RETRAIN_EVERY = 2
MIN_DDM_SAMPLES = 50
MIN_RETRAIN_SAMPLES = 50
MAX_FEEDBACK_BUFFER = 5000
SUSPICIOUS = "Suspicious/Unknown"

FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s", "SYN Flag Count",
    "Packet Length Mean"
]
