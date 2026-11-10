"""数据源适配。每个数据源只负责返回标准 OHLCV 表，不做任何缓存。

标准表：DatetimeIndex（无时区、按日、升序、无重复），列 open/high/low/close/volume，全部 float。
美股 volume 是股数；加密货币 volume 是 USDT 成交额。
"""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", message=".*inplace keyword.*")
from datetime import date, datetime, timedelta, timezone

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


class SymbolNotFound(Exception):
    """数据源里没有这个代码。"""


class SourceError(Exception):
    """数据源暂时不可用。"""


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLUMNS].astype(float).copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "date"
    return df.dropna(subset=["close"])


class Source(ABC):
    market: str

    @abstractmethod
    def fetch(self, code: str, start: date | None) -> pd.DataFrame:
        """返回 start（含）之后的所有日线。start 为 None 时返回全部历史。"""

    @abstractmethod
    def exists(self, code: str) -> bool:
        """代码在数据源里是否存在。"""


class YFinanceSource(Source):
    market = "us"

    def fetch(self, code: str, start: date | None) -> pd.DataFrame:
        import yfinance as yf

        import logging

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        try:
            df = yf.download(
                code,
                start=start.isoformat() if start else "1990-01-01",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as e:  # noqa: BLE001
            raise SourceError(f"yfinance 请求失败：{e}") from e
        if df is None or df.empty:
            if start is None:
                raise SymbolNotFound(f"yfinance 没有找到 {code}")
            return empty_frame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        return _standardize(df)

    def exists(self, code: str) -> bool:
        import yfinance as yf

        try:
            hist = yf.Ticker(code).history(period="5d", auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            raise SourceError(f"yfinance 请求失败：{e}") from e
        return hist is not None and not hist.empty


class BinanceSource(Source):
    market = "crypto"

    def __init__(self):
        self._ex = None

    @property
    def ex(self):
        if self._ex is None:
            import ccxt

            self._ex = ccxt.binance({"enableRateLimit": True})
        return self._ex

    def fetch(self, code: str, start: date | None) -> pd.DataFrame:
        import ccxt

        since = int(datetime.combine(start or date(2017, 1, 1), datetime.min.time(), timezone.utc).timestamp() * 1000)
        rows: list[list] = []
        try:
            while True:
                batch = self.ex.fetch_ohlcv(code, "1d", since=since, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                last = batch[-1][0]
                if len(batch) < 1000:
                    break
                since = last + 86_400_000
                time.sleep(self.ex.rateLimit / 1000)
        except ccxt.BadSymbol as e:
            raise SymbolNotFound(f"币安没有 {code} 这个交易对") from e
        except ccxt.BaseError as e:
            raise SourceError(f"币安请求失败：{e}") from e
        if not rows:
            if start is None:
                raise SymbolNotFound(f"币安没有 {code} 的历史数据")
            return empty_frame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "base_volume"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        # 成交额用 USDT 计价：基础币成交量 × 当日典型价格（币安日线接口不直接给成交额）
        typical = (df["high"] + df["low"] + df["close"]) / 3
        df["volume"] = df["base_volume"] * typical
        df = df.set_index("date")
        # 最后一根如果是今天（UTC），尚未收盘，去掉
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        df = _standardize(df)
        return df[df.index < today]

    def exists(self, code: str) -> bool:
        import ccxt

        try:
            markets = self.ex.load_markets()
        except ccxt.BaseError as e:
            raise SourceError(f"币安请求失败：{e}") from e
        m = markets.get(code)
        return bool(m and m.get("spot") and m.get("active", True))


def default_sources() -> dict[str, Source]:
    import os

    if os.environ.get("WYCKOFF_FAKE_DATA"):
        from .fake import fake_sources

        return fake_sources()
    return {"us": YFinanceSource(), "crypto": BinanceSource()}


def next_day(d: pd.Timestamp) -> date:
    return (d + timedelta(days=1)).date()
