import pandas as pd
import numpy as np


def sma(df: pd.DataFrame, period: int = 10) -> np.ndarray:
    hilo = (df["High"] + df["Low"]) / 2
    return df["Close"].rolling(period, min_periods=1).mean() / hilo
