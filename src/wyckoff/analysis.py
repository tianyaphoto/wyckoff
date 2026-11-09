"""把数据、识别、统计串起来，产出报告和页面需要的结构化结果。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pandas as pd

from .backtest.stats import event_group_for_lookup, load_stats, stats_lookup
from .config import MARKET_NAMES, Market, Params, Paths, load_params
from .data import DataQuality, DataService, check_quality
from .detector import DetectionResult, Event, detect
from .detector.events import CONFIDENCE_ZH, STRUCTURE_ZH, phase_name


@dataclass
class EventView:
    event: Event
    stats: dict | None  # 该事件分组在回测标的池上的统计（可能为空）

    @property
    def h20(self) -> dict | None:
        if not self.stats:
            return None
        return (self.stats.get("horizons") or {}).get("20")

    @property
    def verdict(self) -> str:
        return self.stats["verdict"] if self.stats else "暂无统计"


@dataclass
class SymbolAnalysis:
    market: Market
    code: str
    df: pd.DataFrame
    result: DetectionResult
    quality: DataQuality
    params: Params
    current: dict
    recent_events: list[EventView] = field(default_factory=list)
    generated_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.now().floor("s"))

    @property
    def market_zh(self) -> str:
        return MARKET_NAMES[self.market]

    @property
    def last_date(self) -> str | None:
        return self.df.index[-1].date().isoformat() if not self.df.empty else None

    @property
    def last_close(self) -> float | None:
        return float(self.df["close"].iloc[-1]) if not self.df.empty else None

    @property
    def change_pct(self) -> float | None:
        if len(self.df) < 2:
            return None
        return float(self.df["close"].iloc[-1] / self.df["close"].iloc[-2] - 1)

    @property
    def phase_zh(self) -> str | None:
        ph = self.current.get("phase")
        return f"{ph} {phase_name(self.current.get('kind'), ph)}" if ph else None

    @property
    def kind_zh(self) -> str | None:
        k = self.current.get("kind")
        return STRUCTURE_ZH[k] if k else None

    @property
    def confidence_zh(self) -> str | None:
        c = self.current.get("confidence")
        return CONFIDENCE_ZH[c] if c else None

    @property
    def latest_event(self) -> Event | None:
        return self.current.get("latest_event")

    @property
    def headline(self) -> str:
        """报告第一句话。"""
        cur = self.current
        name = f"{self.code} 日线"
        if cur.get("phase"):
            s = f"{name}：{self.kind_zh} {self.phase_zh}（可信度{self.confidence_zh}）"
            if cur.get("candidate") and cur["phase"] == "A":
                s += f"，出现候选 {cur['candidate']}，等待 AR 确认"
            elif self.latest_event is not None:
                s += f"，{self._days_ago(self.latest_event)}确认 {self.latest_event.label}"
            return s
        s = f"{name}：当前无威克夫结构"
        if cur.get("trend"):
            s += f"，{cur['trend']}"
        if self.latest_event is not None:
            s += f"；最近一次事件是 {self.latest_event.label}，{self._days_ago(self.latest_event)}确认"
        return s

    def _days_ago(self, e: Event) -> str:
        d = (self.df.index[-1] - e.confirmed).days
        return "今天" if d == 0 else f"{d} 天前"

    @property
    def active_structure(self):
        sid = self.current.get("structure_id")
        for s in self.result.structures:
            if s.end_reason == "ongoing":
                return s
        return None

    def to_overview_row(self) -> dict:
        le = self.latest_event
        return {
            "market": self.market,
            "market_zh": self.market_zh,
            "code": self.code,
            "last_date": self.last_date,
            "last_close": self.last_close,
            "change_pct": self.change_pct,
            "kind": self.current.get("kind"),
            "kind_zh": self.kind_zh,
            "phase": self.current.get("phase"),
            "phase_zh": self.phase_zh,
            "confidence": self.current.get("confidence"),
            "confidence_zh": self.confidence_zh,
            "trend": self.current.get("trend"),
            "candidate": self.current.get("candidate"),
            "latest_event": le.label if le else None,
            "latest_event_kind": le.kind if le else None,
            "latest_event_days": self.current.get("days_since_event"),
            "latest_event_date": le.confirmed.date().isoformat() if le else None,
            "warnings": self.quality.warnings,
            "bars": len(self.df),
        }


class Analyzer:
    """带缓存的分析服务：同一标的、同一数据版本只算一次。"""

    def __init__(self, paths: Paths | None = None, service: DataService | None = None):
        self.paths = paths or Paths()
        self.service = service or DataService(self.paths)
        self._cache: dict[tuple, SymbolAnalysis] = {}
        self._lock = threading.Lock()
        self._stats_cache: tuple[float, dict] | None = None

    # ---- 统计
    def stats_json(self) -> dict:
        p = self.paths.stats
        mtime = p.stat().st_mtime if p.exists() else 0.0
        if self._stats_cache is None or self._stats_cache[0] != mtime:
            self._stats_cache = (mtime, load_stats(p))
        return self._stats_cache[1]

    def invalidate(self, market: Market | None = None, code: str | None = None) -> None:
        with self._lock:
            if market is None:
                self._cache.clear()
            else:
                for k in list(self._cache):
                    if k[0] == market and k[1] == code:
                        del self._cache[k]

    # ---- 分析
    def analyze(self, market: Market, code: str, refresh: bool = False, recent_days: int = 365) -> SymbolAnalysis:
        df = self.service.load(market, code, refresh=refresh)
        version = (market, code, len(df), str(df.index[-1]) if not df.empty else None, self._stats_mtime())
        with self._lock:
            hit = self._cache.get(version)
        if hit is not None:
            return hit
        params = load_params(market, self.paths)
        result = detect(df, params)
        quality = check_quality(df, params.min_history_bars)
        lookup = stats_lookup(self.stats_json(), market)
        cutoff = df.index[-1] - pd.Timedelta(days=recent_days) if not df.empty else None
        recent = [
            EventView(e, lookup.get(event_group_for_lookup(e) or ""))
            for e in reversed(result.events)
            if cutoff is None or e.confirmed >= cutoff
        ]
        sa = SymbolAnalysis(
            market=market,
            code=code,
            df=df,
            result=result,
            quality=quality,
            params=params,
            current=result.current(),
            recent_events=recent,
        )
        with self._lock:
            self._cache[version] = sa
        return sa

    def _stats_mtime(self) -> float:
        p = self.paths.stats
        return p.stat().st_mtime if p.exists() else 0.0

    def analyze_many(self, items: list[tuple[Market, str]], refresh: bool = False) -> list[SymbolAnalysis]:
        out = []
        for m, c in items:
            try:
                out.append(self.analyze(m, c, refresh=refresh))
            except Exception as e:  # noqa: BLE001
                out.append(_failed_analysis(m, c, str(e)))
        return out


def _failed_analysis(market: Market, code: str, error: str) -> SymbolAnalysis:
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    q = DataQuality(bars=0, first=None, last=None, warnings=[f"加载失败：{error}"])
    res = detect(empty, load_params(market)) if False else None  # noqa: SIM108
    from .detector.engine import DetectionResult

    res = DetectionResult(events=[], structures=[], phases=pd.DataFrame(), frame=empty)
    return SymbolAnalysis(
        market=market,
        code=code,
        df=empty,
        result=res,
        quality=q,
        params=load_params(market),
        current={"phase": None, "kind": None, "confidence": None, "trend": None, "latest_event": None},
    )
