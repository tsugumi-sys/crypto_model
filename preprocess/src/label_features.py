from typing import Tuple
import pandas as pd
import numpy as np
import numba
from scipy.ndimage.interpolation import shift


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = np.max(ranges, axis=1)
    atr = tr.rolling(period).sum() / period
    return atr


def position_price_by_atr(df: pd.DataFrame, pips: float) -> np.ndarray:
    if pips <= 0:
        raise ValueError(f"Invalid 'pips' valuse {pips}. 'pips' shoud be more than 0")

    _atr = atr(df, 14)
    limit_price_dist = _atr * 0.5
    limit_price_dist = np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips

    # return (Buy_Price, Sell_Price)
    return df["Close"] - limit_price_dist, df["Close"] + limit_price_dist


@numba.njit
def buy_calc_force_entry_price(buy_price: np.ndarray, low_price: np.ndarray, pips: float) -> Tuple[np.ndarray, np.ndarray]:
    y = buy_price.copy()
    y[:] = np.nan
    force_entry_time = buy_price.copy()
    force_entry_time[:] = np.nan
    for i in range(buy_price.size):
        for j in range(i + 1, buy_price.size):
            if round(low_price[j] / pips) < round(buy_price[j - 1] / pips):
                y[i] = buy_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


@numba.njit
def sell_calc_force_entry_price(sell_price: np.ndarray, high_price: np.ndarray, pips: float) -> Tuple[np.ndarray, np.ndarray]:
    y = sell_price.copy()
    y[:] = np.nan
    force_entry_time = sell_price.copy()
    force_entry_time[:] = np.nan

    for i in range(sell_price.size):
        for j in range(i + 1, sell_price.size):
            if round(high_price[j] / pips) > round(sell_price[j - 1] / pips):
                y[i] = sell_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


def position_label(
    df: pd.DataFrame, pips: float = 1.0, fee_percent: float = 0.01, horizon_barrier: int = 1, executionType: str = "limit"
) -> Tuple[np.ndarray, np.ndarray]:
    if pips <= 0:
        raise ValueError(f"Invalid 'pips' value {pips}. 'pips' shoud be more than 0")

    if executionType not in ["limit", "stop", "check"]:
        raise ValueError(f"Invalid 'executionType' value {executionType}. 'executionType' shoud be in ['limit', 'stop']")

    df["fee"] = fee_percent / 100
    buy_price, sell_price = position_price_by_atr(df, pips)
    buy_executed = ((buy_price / pips).round() > (df["Low"].shift(-1) / pips).round()).astype("float64")
    sell_executed = ((sell_price / pips).round() < (df["High"].shift(-1) / pips).round()).astype("float64")
    buy_force_entry_price, buy_force_entry_time = buy_calc_force_entry_price(buy_price.values, df["Low"].values, pips)
    sell_force_entry_price, sell_force_entry_time = sell_calc_force_entry_price(sell_price.values, df["High"].values, pips)

    y_buy = np.where(
        buy_executed,
        shift(sell_force_entry_price, -horizon_barrier, cval=np.NaN, order=0) / buy_price - 1 - 2 * df["fee"],
        0,
    )
    buy_cost = np.where(
        buy_executed,
        buy_price / df["Close"] - 1 + df["fee"],
        0,
    )

    y_sell = np.where(
        sell_executed,
        (shift(buy_force_entry_price, -horizon_barrier, cval=np.NaN, order=0) / sell_price - 1) * (-1) - 2 * df["fee"],
        0,
    )
    sell_cost = np.where(
        sell_executed,
        (sell_price / df["Close"] - 1) * (-1) + df["fee"],
        0,
    )

    if executionType == "limit":
        return y_buy, buy_cost, y_sell, sell_cost
    elif executionType == "stop":
        return y_sell, sell_cost, y_buy, buy_cost
    else:
        return y_buy, buy_cost, y_sell, sell_cost, buy_executed, sell_executed, buy_force_entry_time, sell_force_entry_time


def sample_target(df: pd.DataFrame):
    _return = np.log((df["Close"] + 1) / (df["Close"].shift(-1) + 1))
    return np.where(_return > 0, 1, 0)
