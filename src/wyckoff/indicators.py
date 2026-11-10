"""指标：ATR、量比、均线、N 日极值。全部只用当天及以前的数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Params


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    # Wilder 平滑
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    out["atr"] = atr(out, p.atr_period)
    # 量比：当日成交量 ÷ 过去 N 日平均成交量（不含当天）
    avg_vol = out["volume"].shift(1).rolling(p.volume_period, min_periods=p.volume_period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vol_ratio"] = (out["volume"] / avg_vol).replace([np.inf, -np.inf], np.nan)
    out["sma_trend"] = out["close"].rolling(p.trend_sma, min_periods=p.trend_sma).mean()
    out["sma200"] = out["close"].rolling(200, min_periods=200).mean()
    # 过去 N 日极值（不含当天）
    out["low_n"] = out["low"].shift(1).rolling(p.extreme_lookback, min_periods=p.extreme_lookback).min()
    out["high_n"] = out["high"].shift(1).rolling(p.extreme_lookback, min_periods=p.extreme_lookback).max()
    out["range"] = out["high"] - out["low"]
    # 长基线（可选）：过去 N 日的平均真实波幅和中位成交量，都不含当天。
    # 持续数周的暴跌会把 14 日 ATR 和 20 日均量一起抬高，长基线用来在这种行情里仍能认出高潮日。
    if p.long_baseline_bars > 0:
        nb = p.long_baseline_bars
        prev_close = out["close"].shift(1)
        tr = pd.concat(
            [out["range"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1
        ).max(axis=1)
        out["atr_long"] = tr.shift(1).rolling(nb, min_periods=nb).mean()
        med_vol = out["volume"].shift(1).rolling(nb, min_periods=nb).median()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vol_ratio_long"] = (out["volume"] / med_vol).replace([np.inf, -np.inf], np.nan)
    else:
        out["atr_long"] = np.nan
        out["vol_ratio_long"] = np.nan
    return out
