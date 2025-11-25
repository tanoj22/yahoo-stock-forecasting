# src/models_arima.py

import numpy as np
import pandas as pd
import pickle
from typing import Optional
from pmdarima import auto_arima

from .config import MODEL_DIR, TARGET_COL


def train_arima(
    train_df: pd.DataFrame,
    seasonal: bool = False,
    m: int = 1,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    max_P: int = 2,
    max_Q: int = 2,
    max_D: int = 1,
    trace: bool = True,
):
    """
    Train an ARIMA/SARIMA model using pmdarima.auto_arima on the TARGET_COL.
    """
    y_train = train_df[TARGET_COL].values

    model = auto_arima(
        y_train,
        seasonal=seasonal,
        m=m,                    # season length (1 for no seasonality in daily stock)
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        max_P=max_P,
        max_Q=max_Q,
        max_D=max_D,
        trace=trace,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        information_criterion="aic",
    )
    return model


def forecast_arima(model, n_periods: int) -> np.ndarray:
    """
    Forecast n_periods ahead using fitted pmdarima model.
    """
    return model.predict(n_periods=n_periods)


def save_arima_model(model, filename: str = "arima_model.pkl") -> str:
    """
    Save fitted ARIMA model to MODEL_DIR and return path.
    """
    path = f"{MODEL_DIR}/{filename}"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_arima_model(filename: str = "arima_model.pkl"):
    """
    Load ARIMA model from pickle.
    NOTE: pmdarima must be installed to unpickle successfully.
    """
    path = f"{MODEL_DIR}/{filename}"
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
