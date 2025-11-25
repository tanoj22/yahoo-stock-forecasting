# src/config.py

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "yahoo_stock.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Target and features
DATE_COL = "Date"
TARGET_COL = "Close"

# Train/val/test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15   # test will be 1 - TRAIN - VAL

# LSTM config
LSTM_LOOKBACK = 60       # number of past days used as input
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2
LSTM_BATCH_SIZE = 64
LSTM_NUM_EPOCHS = 50
LSTM_LR = 1e-3

# TFT config (you can tweak)
TFT_MAX_ENCODER_LENGTH = 60
TFT_MAX_PREDICTION_LENGTH = 1
TFT_BATCH_SIZE = 64
TFT_MAX_EPOCHS = 50

RANDOM_SEED = 42
