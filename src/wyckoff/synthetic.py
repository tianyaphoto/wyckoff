"""合成行情：随机游走和手工搭建的威克夫场景。测试和安慰剂检验都用它。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def random_walk(
    n: int = 600,
    seed: int = 0,
    start: float = 100.0,
    vol: float = 0.015,
    drift: float = 0.0,
    vol_sigma: float = 0.3,
    tails: float | None = None,
) -> pd.DataFrame:
    """几何随机游走日线，成交量围绕 1e6 波动。tails 给定自由度时收益用 t 分布（肥尾），更像真实市场。"""
    rng = np.random.default_rng(seed)
    if tails:
        rets = drift + vol * rng.standard_t(tails, n) / np.sqrt(tails / (tails - 2))
    else:
        rets = rng.normal(drift, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[start, close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    wick = np.abs(rets) * 0.5 + vol / 2
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 1, n)) * wick)
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 1, n)) * wick)
    # 成交量和当日波动相关：大波动日放量，这是真实市场的基本特征
    volume = 1e6 * np.exp(rng.normal(0, vol_sigma, n) + 0.8 * np.abs(rets) / vol - 0.8)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close, "volume": volume}, index=idx)


def active_market(n: int = 2000, seed: int = 0, **kw) -> pd.DataFrame:
    """肥尾、量价相关的随机行情，事件密度接近真实市场。没有任何真实优势，用作安慰剂。"""
    kw.setdefault("vol_sigma", 0.5)
    kw.setdefault("tails", 3)
    return random_walk(n, seed=seed, **kw)


class ScenarioBuilder:
    """逐根 K 线搭场景。价格以“点”为单位，默认每根 K 线振幅约 1 点，成交量 100 万。

    先 calm() 铺出足够的预热数据，让 ATR 稳定在 1 左右、量比平均在 1 左右。
    """

    def __init__(self, start_price: float = 100.0, start_date: str = "2020-01-01"):
        self.rows: list[tuple[float, float, float, float, float]] = []
        self.price = start_price
        self.start_date = start_date

    def bar(self, open_: float, high: float, low: float, close: float, volume: float = 1e6):
        assert low <= min(open_, close) and high >= max(open_, close), "K 线高低价不合法"
        self.rows.append((open_, high, low, close, volume))
        self.price = close
        return self

    def calm(self, n: int, drift: float = 0.0, amplitude: float = 1.0, volume: float = 1e6):
        """平稳横盘：每根振幅 amplitude，收盘在开盘附近交替。"""
        for k in range(n):
            o = self.price
            c = o + drift + (0.2 if k % 2 == 0 else -0.2)
            self.bar(o, max(o, c) + amplitude / 2, min(o, c) - amplitude / 2, c, volume)
        return self

    def trend(self, n: int, step: float, amplitude: float = 1.0, volume: float = 1e6):
        for _ in range(n):
            o = self.price
            c = o + step
            self.bar(o, max(o, c) + amplitude / 2, min(o, c) - amplitude / 2, c, volume)
        return self

    def frame(self) -> pd.DataFrame:
        idx = pd.bdate_range(self.start_date, periods=len(self.rows))
        return pd.DataFrame(self.rows, columns=["open", "high", "low", "close", "volume"], index=idx)


def accumulation_scenario(
    *,
    climax_volume: float = 3e6,
    climax_range: float = 3.0,
    range_bars: int = 20,
    spring_depth: float = 0.8,
    spring_volume: float = 0.6e6,
    with_spring: bool = True,
    with_sos: bool = True,
    sos_volume: float = 2e6,
) -> tuple[pd.DataFrame, dict]:
    """一段教科书式吸筹：下跌 → SC → AR → 区间 → Spring → SOS。

    返回 (数据, 关键日期索引)。ATR 在预热后约等于 1。
    """
    b = ScenarioBuilder(100.0)
    b.calm(60)  # 预热
    b.trend(30, -0.4)  # 下跌趋势，收盘远低于 50 日均线
    marks: dict[str, int] = {}
    # SC：大振幅、放量、创新低
    o = b.price
    sc_low = o - climax_range
    marks["SC"] = len(b.rows)
    b.bar(o, o + 0.2, sc_low, sc_low + 0.6, climax_volume)
    # AR：5 根上涨，反弹约 3 点
    for k in range(5):
        o = b.price
        c = o + 0.6
        if k == 4:
            marks["AR"] = len(b.rows)
        b.bar(o, c + 0.5, o - 0.3, c, 1.2e6)
    ar_high = max(r[1] for r in b.rows[marks["AR"] : marks["AR"] + 1])
    # 回落 ≥ 1 ATR 确认 AR
    for _ in range(3):
        o = b.price
        b.bar(o, o + 0.3, o - 0.8, o - 0.5, 0.9e6)
    marks["AR_confirmed"] = len(b.rows) - 1
    # 区间震荡
    mid = (sc_low + ar_high) / 2
    b.price = mid
    for k in range(range_bars):
        o = b.price
        c = mid + (0.4 if k % 2 == 0 else -0.4)
        b.bar(o, max(o, c) + 0.5, min(o, c) - 0.5, c, 0.8e6)
    if with_spring:
        marks["SPRING"] = len(b.rows)
        o = b.price
        b.bar(o, o + 0.2, sc_low - spring_depth, sc_low + 0.3, spring_volume)  # 刺破下沿当天收回
        for _ in range(3):
            o = b.price
            b.bar(o, o + 0.9, o - 0.2, o + 0.7, 1.0e6)
    if with_sos:
        # 先回到区间上部
        while b.price < ar_high - 0.5:
            o = b.price
            b.bar(o, o + 0.9, o - 0.2, o + 0.6, 1.0e6)
        marks["SOS"] = len(b.rows)
        o = b.price
        c = ar_high + 1.2
        b.bar(o, c + 0.3, o - 0.2, c, sos_volume)
        b.trend(10, 0.4, volume=1.1e6)
    marks["sc_low"] = sc_low
    marks["ar_high"] = ar_high
    return b.frame(), marks


def mirror(df: pd.DataFrame, pivot: float | None = None) -> pd.DataFrame:
    """把价格上下翻转：p -> pivot*2 - p。high/low 互换。成交量不变。"""
    if pivot is None:
        pivot = float(df["close"].iloc[0])
    out = df.copy()
    out["open"] = 2 * pivot - df["open"]
    out["close"] = 2 * pivot - df["close"]
    out["high"] = 2 * pivot - df["low"]
    out["low"] = 2 * pivot - df["high"]
    return out
