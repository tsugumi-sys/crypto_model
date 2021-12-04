from typing import Tuple
import pandas as pd
import numpy as np
import numba


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tr = np.max(ranges, axis=1)
    atr = tr.rolling(period).sum() / period
    return atr


def position_price_by_atr(df: pd.DataFrame, pips: float = 1.0, position: str = "buy") -> np.ndarray:
    if pips <= 0:
        raise ValueError(f"Invalid 'pips' valuse {pips}. 'pips' shoud be more than 0")
    if position not in ["buy", "sell"]:
        raise ValueError(f"Invalid position 'value' {position}. position should be 'buy' or 'sell'")

    _atr = atr(df, 14)
    limit_price_dist = _atr * 0.5
    limit_price_dist = np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips
    if position == "buy":
        return df["Close"] - limit_price_dist
    else:
        return df["Close"] + limit_price_dist


@numba.njit
def buy_calc_force_entry_price(entry_price: np.ndarray, boundary: np.ndarray, pips: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    y = entry_price.copy()
    y[:] = np.nan
    force_entry_time = entry_price.copy()
    force_entry_time[:] = np.nan
    for i in range(entry_price.size):
        for j in range(i + 1, entry_price.size):
            if round(boundary[j] / pips) < round(entry_price[j - 1] / pips):
                y[i] = entry_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


@numba.njit
def sell_calc_force_entry_price(entry_price: np.ndarray, boundary: np.ndarray, pips: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    y = entry_price.copy()
    y[:] = np.nan
    force_entry_time = entry_price.copy()
    force_entry_time[:] = np.nan

    for i in range(entry_price.size):
        for j in range(i + 1, entry_price.size):
            if round(boundary[j] / pips) > round(entry_price[j - 1] / pips):
                y[i] = entry_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


def position_force_entry_price(df: pd.DataFrame, pips: float = 1.0, position: str = "buy") -> Tuple[np.ndarray, np.ndarray]:
    if pips <= 0:
        raise ValueError(f"Invalid 'pips' valuse {pips}. 'pips' shoud be more than 0")
    if position not in ["buy", "sell"]:
        raise ValueError(f"Invalid position 'value' {position}. position should be 'buy' or 'sell'")

    if position == "buy":
        buy_price = position_price_by_atr(df, pips, "buy")
        buy_force_entry_price, buy_force_entry_time = buy_calc_force_entry_price(buy_price.values, df["Low"].values, pips)
        return buy_force_entry_price, buy_force_entry_time
    else:
        sell_price = position_price_by_atr(df, pips, "sell")
        sell_force_entry_price, sell_force_entry_time = sell_calc_force_entry_price(sell_price.values, df["High"].values, pips)
        return sell_force_entry_price, sell_force_entry_time


def position_label(df: pd.DataFrame, pips: float = 1.0, fee: float = 0.01, horizon_barrier: int = 1, position: str = "buy"):
    if pips <= 0:
        raise ValueError(f"Invalid 'pips' valuse {pips}. 'pips' shoud be more than 0")
    if position not in ["buy", "sell"]:
        raise ValueError(f"Invalid position 'value' {position}. position should be 'buy' or 'sell'")

    if position == "buy":
        df["fee"] = fee
        buy_price = position_price_by_atr(df, pips, position)
        buy_executed = ((buy_price / pips).round() > (df["Low"].shift(-1) / pips).round()).astype("float64")
        sell_force_entry_price, _ = position_force_entry_price(df, pips, "sell")
        y_buy = np.where(
            buy_executed,
            sell_force_entry_price.shift(-horizon_barrier) / buy_price - 1 - 2 * df["fee"],
            0,
        )
        buy_cost = np.where(
            buy_executed,
            buy_price / df["Close"] - 1 + fee,
            0,
        )
        return y_buy, buy_cost
    else:
        df["fee"] = fee
        sell_price = position_price_by_atr(df, pips, position)
        sell_executed = ((sell_price / pips).round() < (df["High"].shift(-1) / pips).round()).astype("float64")
        buy_force_entry_price, _ = position_force_entry_price(df, pips, "buy")
        y_sell = np.where(
            sell_executed,
            (buy_force_entry_price.shift(-horizon_barrier) / sell_price - 1) * (-1) - 2 * df["fee"],
            0,
        )
        sell_cost = np.where(
            sell_executed,
            (sell_price / df["Close"] - 1) * (-1) + fee,
            0,
        )
        return y_sell, sell_cost


def sample_target(df: pd.DataFrame):
    _return = np.log((df["Close"] + 1) / (df["Close"].shift(-1) + 1))
    return np.where(_return > 0, 1, 0)
