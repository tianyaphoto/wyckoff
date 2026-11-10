"""合成数据源：离线演示和测试用，不访问网络。

设置环境变量 WYCKOFF_FAKE_DATA=1 后，Web 服务和命令行都会用它代替真实数据源。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..synthetic import active_market
from .sources import Source, SourceError, SymbolNotFound, empty_frame

UNKNOWN_CODES = {"NOPE", "ZZZZ", "NOPE/USDT"}
BROKEN_CODES = {"BROKEN", "BROKEN/USDT"}  # 模拟数据源超时


class FakeSource(Source):
    def __init__(self, market: str, bars: int = 1500, today: date | None = None):
        self.market = market
        self.bars = bars
        self.today = today  # 模拟“今天”，用来测试增量更新
        self.calls: list[tuple[str, date | None]] = []

    def _full(self, code: str) -> pd.DataFrame:
        seed = sum(ord(c) for c in code) % 1000
        df = active_market(self.bars, seed=seed, start=50 + seed % 400, vol=0.02 if self.market == "us" else 0.04)
        if self.today is not None:
            df = df[df.index.date <= self.today]
        return df

    def fetch(self, code: str, start: date | None) -> pd.DataFrame:
        self.calls.append((code, start))
        if code in UNKNOWN_CODES:
            raise SymbolNotFound(f"没有找到 {code}")
        if code in BROKEN_CODES:
            raise SourceError("模拟的数据源超时")
        df = self._full(code)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        return df if not df.empty else empty_frame()

    def exists(self, code: str) -> bool:
        if code in UNKNOWN_CODES:
            return False
        return True


def fake_sources(bars: int = 1500) -> dict[str, Source]:
    return {"us": FakeSource("us", bars), "crypto": FakeSource("crypto", bars)}
