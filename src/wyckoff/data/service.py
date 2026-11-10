"""数据服务：缓存 + 增量更新 + 数据检查。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import Market, Paths
from .sources import COLUMNS, Source, SymbolNotFound, default_sources, empty_frame, next_day

__all__ = ["DataService", "DataQuality", "SymbolNotFound", "check_quality"]


@dataclass
class DataQuality:
    bars: int
    first: str | None
    last: str | None
    gaps: int = 0  # 超过 7 个自然日的空档数（美股节假日不算）
    zero_volume_days: int = 0
    suspicious_jumps: int = 0  # 单日收盘变动超过 50% 的天数
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def check_quality(df: pd.DataFrame, min_bars: int = 200) -> DataQuality:
    q = DataQuality(bars=len(df), first=None, last=None)
    if df.empty:
        q.warnings.append("没有数据")
        return q
    q.first = df.index[0].date().isoformat()
    q.last = df.index[-1].date().isoformat()
    if len(df) < min_bars:
        q.warnings.append(f"历史只有 {len(df)} 根 K 线，少于 {min_bars} 根，结果仅供参考")
    diffs = df.index.to_series().diff().dt.days.iloc[1:]
    q.gaps = int((diffs > 7).sum())
    if q.gaps:
        q.warnings.append(f"有 {q.gaps} 处超过一周的数据空档")
    q.zero_volume_days = int((df["volume"] <= 0).sum())
    if q.zero_volume_days:
        q.warnings.append(f"有 {q.zero_volume_days} 天成交量为零")
    jumps = df["close"].pct_change().abs() > 0.5
    q.suspicious_jumps = int(jumps.sum())
    if q.suspicious_jumps:
        q.warnings.append(f"有 {q.suspicious_jumps} 天单日涨跌超过 50%，可能是未复权或数据错误")
    return q


class DataService:
    """按标的缓存日线到 Parquet；update 只拉最后一根之后的数据。"""

    def __init__(self, paths: Paths | None = None, sources: dict[str, Source] | None = None):
        self.paths = paths or Paths()
        self.sources = sources or default_sources()

    # ---- 路径
    def cache_path(self, market: Market, code: str) -> Path:
        safe = code.replace("/", "_")
        return self.paths.data / market / f"{safe}.parquet"

    def has_cache(self, market: Market, code: str) -> bool:
        return self.cache_path(market, code).exists()

    def read(self, market: Market, code: str) -> pd.DataFrame:
        p = self.cache_path(market, code)
        if not p.exists():
            return empty_frame()
        df = pd.read_parquet(p)
        df.index = pd.DatetimeIndex(df.index)
        return df

    def write(self, market: Market, code: str, df: pd.DataFrame) -> None:
        p = self.cache_path(market, code)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)

    # ---- 更新
    def update(self, market: Market, code: str, start: date | str | None = None, force: bool = False) -> pd.DataFrame:
        """增量更新并返回完整数据。force=True 时全量重拉。"""
        source = self.sources[market]
        if isinstance(start, str):
            start = date.fromisoformat(start)
        cached = empty_frame() if force else self.read(market, code)
        if cached.empty:
            fresh = source.fetch(code, start)
            if fresh.empty:
                raise SymbolNotFound(f"{code} 没有返回任何数据")
            merged = fresh
        else:
            # 重拉最后 3 根，覆盖掉数据源对最近几天的修正
            last = cached.index[-1]
            refetch_from = (last - pd.Timedelta(days=5)).date()
            fresh = source.fetch(code, refetch_from)
            if fresh.empty:
                return cached
            merged = pd.concat([cached[cached.index < fresh.index[0]], fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if start is not None:
            merged = merged[merged.index >= pd.Timestamp(start)]
        if not merged.empty:
            self.write(market, code, merged)
        return merged

    def load(self, market: Market, code: str, refresh: bool = False) -> pd.DataFrame:
        """读缓存；没有缓存或要求刷新时去数据源取。"""
        if refresh or not self.has_cache(market, code):
            return self.update(market, code)
        return self.read(market, code)

    def exists(self, market: Market, code: str) -> bool:
        return self.sources[market].exists(code)

    def delete_cache(self, market: Market, code: str) -> None:
        p = self.cache_path(market, code)
        if p.exists():
            p.unlink()

    def last_updated(self, market: Market, code: str) -> str | None:
        df = self.read(market, code)
        return df.index[-1].date().isoformat() if not df.empty else None

    @staticmethod
    def incremental_start(df: pd.DataFrame) -> date | None:
        return next_day(df.index[-1]) if not df.empty else None
