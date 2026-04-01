"""
config.py - Central configuration for the Sarcasm Detection System
"""

import os

# ─── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ─── Dataset ───────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(DATA_DIR, "sample_dataset.csv")
REQUIRED_COLUMNS = ["text", "context", "label", "emotion", "language"]

# ─── Model Paths ───────────────────────────────────────────────────────────────
SARCASM_MODEL_PATH = os.path.join(MODELS_DIR, "sarcasm_model.pkl")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pkl")

# ─── Model Settings ────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQUENCE_LENGTH = 128
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ─── Supported Languages ───────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
}
DEFAULT_LANGUAGE = "en"

# ─── Emotion Labels ─────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Happy", "Angry", "Sad", "Neutral"]

# ─── Sarcasm Labels ─────────────────────────────────────────────────────────────
SARCASM_LABELS = {0: "Not Sarcastic", 1: "Sarcastic"}

# ─── UI Settings ────────────────────────────────────────────────────────────────
APP_TITLE = "Context-Aware Sarcasm Detection System"
APP_SUBTITLE = "Multilingual · Explainable AI · Emotion-Aware"
APP_ICON = "🎭"
APP_VERSION = "1.0.0"

# ─── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(LOGS_DIR, "sarcasm_detection.log")
LOG_LEVEL = "INFO"

# ─── Create directories if missing ─────────────────────────────────────────────
for _dir in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
