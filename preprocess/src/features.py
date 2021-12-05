import pandas as pd
import numpy as np


def hilo(df: pd.DataFrame):
    return (df["High"] + df["Low"]) / 2


def sma(df: pd.DataFrame, period: int = 10) -> np.ndarray:
    return df["Close"].rolling(period, min_periods=1).mean() / hilo(df)


def std(df: pd.DataFrame, period: int = 10) -> np.ndarray:
    return df["Close"].rolling(period).std()
