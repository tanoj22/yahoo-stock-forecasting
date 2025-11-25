# src/data_utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
from .config import DATA_PATH, DATE_COL, TARGET_COL, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED

def load_raw_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # standard Yahoo format
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df

def train_val_test_split_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]
    return train, val, test

def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: list,
    target_col: str = TARGET_COL
) -> Dict[str, MinMaxScaler]:
    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()

    feat_scaler.fit(train_df[feature_cols].values)
    targ_scaler.fit(train_df[[target_col]].values)

    return {"feature": feat_scaler, "target": targ_scaler}

def apply_scalers(
    df: pd.DataFrame,
    feature_cols: list,
    scalers: Dict[str, MinMaxScaler],
    target_col: str = TARGET_COL
) -> Tuple[np.ndarray, np.ndarray]:
    X = scalers["feature"].transform(df[feature_cols].values)
    y = scalers["target"].transform(df[[target_col]].values)
    return X, y

def create_lstm_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: (N, num_features), y: (N, 1)
    Returns:
        X_seq: (N-lookback, lookback, num_features)
        y_seq: (N-lookback,)
    """
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).reshape(-1, 1)
    return X_seq, y_seq

def inverse_transform_target(
    y_scaled: np.ndarray,
    target_scaler: MinMaxScaler
) -> np.ndarray:
    return target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
