"""事件统计。

每个事件从确认日收盘价入场，按事件方向（吸筹侧做多、派发侧做空）看之后 5、20、60 根 K 线的收益。
基准 = 同一标的、同一历史区间里任意一天入场、同样方向、同样持有期的收益分布。
超额 = 事件收益 − 该标的基准均值；用 bootstrap 给超额均值 95% 置信区间；区间不跨 0 且样本 ≥ 30 才算显著。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..config import Market, Params, Paths, load_params, load_universe
from ..detector import DetectionResult, Event, detect
from ..detector.events import EVENT_INFO, event_side

HORIZONS = (5, 20, 60)
MIN_SAMPLES = 30
BOOTSTRAP_N = 2000


@dataclass
class EventSample:
    market: str
    symbol: str
    kind: str
    group: str  # 统计分组键，如 SPRING / SPRING·缩量 / AR·吸筹
    structure: str
    occurred: str
    confirmed: str
    side: int
    entry: float
    ret: dict[int, float | None]  # horizon -> 方向调整后的收益
    mae: dict[int, float | None]  # horizon -> 最大不利波动（负数）
    baseline_mean: dict[int, float | None]  # 该标的基准均值


def forward_returns(close: np.ndarray, low: np.ndarray, high: np.ndarray, i: int, side: int, h: int):
    """从第 i 天收盘入场，持有 h 根 K 线。返回 (收益, 最大不利波动)，数据不够返回 (None, None)。"""
    if i + h >= len(close):
        return None, None
    entry = close[i]
    exit_ = close[i + h]
    ret = side * (exit_ / entry - 1)
    path_low = low[i + 1 : i + h + 1]
    path_high = high[i + 1 : i + h + 1]
    if side > 0:
        mae = float(path_low.min() / entry - 1)
    else:
        mae = float(-(path_high.max() / entry - 1))
    return float(ret), float(min(mae, 0.0))


def baseline_means(close: np.ndarray, side: int) -> dict[int, float]:
    """任意一天入场、持有 h 天的平均收益（方向调整后）。"""
    out = {}
    for h in HORIZONS:
        if len(close) <= h:
            out[h] = float("nan")
            continue
        r = side * (close[h:] / close[:-h] - 1)
        out[h] = float(np.mean(r))
    return out


def group_keys(e: Event) -> list[str]:
    keys = []
    if e.kind == "AR":
        keys.append(f"AR·{'吸筹' if e.structure == 'accumulation' else '派发'}")
    elif e.kind == "RANGE":
        return []  # 区间本身不是交易信号
    else:
        keys.append(e.kind)
        if e.kind in ("SPRING", "UTAD") and e.volume_class:
            keys.append(f"{e.kind}·{e.volume_class}")
    return keys


def collect_samples(market: str, symbol: str, df: pd.DataFrame, result: DetectionResult) -> list[EventSample]:
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    idx = {d: i for i, d in enumerate(df.index)}
    base_cache: dict[int, dict[int, float]] = {}
    samples: list[EventSample] = []
    for e in result.events:
        keys = group_keys(e)
        if not keys:
            continue
        side = event_side(e.kind, e.structure)
        if side == 0:
            continue
        i = idx[e.confirmed]
        if side not in base_cache:
            base_cache[side] = baseline_means(close, side)
        rets, maes = {}, {}
        for h in HORIZONS:
            r, m = forward_returns(close, low, high, i, side, h)
            rets[h], maes[h] = r, m
        for g in keys:
            samples.append(
                EventSample(
                    market=market,
                    symbol=symbol,
                    kind=e.kind,
                    group=g,
                    structure=e.structure,
                    occurred=e.occurred.date().isoformat(),
                    confirmed=e.confirmed.date().isoformat(),
                    side=side,
                    entry=float(close[i]),
                    ret=rets,
                    mae=maes,
                    baseline_mean=base_cache[side],
                )
            )
    return samples


@dataclass
class GroupStats:
    market: str
    group: str
    kind: str
    side: int
    n: int
    horizons: dict[int, dict] = field(default_factory=dict)
    # 每个 horizon: n, win_rate, mean, median, mae_mean, baseline_mean, excess_mean, ci_low, ci_high, significant
    verdict: str = "样本不足"  # 显著优于基准 / 不显著 / 样本不足（以 20 日持有期为准）
    significant_horizons: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["horizons"] = {str(k): v for k, v in self.horizons.items()}
        return d


def _bootstrap_ci(x: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(samples: Iterable[EventSample]) -> list[GroupStats]:
    df = pd.DataFrame([asdict(s) for s in samples])
    out: list[GroupStats] = []
    if df.empty:
        return out
    for (market, group), g in df.groupby(["market", "group"], sort=True):
        kind = g["kind"].iloc[0]
        side = int(g["side"].iloc[0])
        gs = GroupStats(market=market, group=group, kind=kind, side=side, n=len(g))
        for h in HORIZONS:
            r = np.array([x[h] for x in g["ret"] if x[h] is not None], dtype=float)
            m = np.array([x[h] for x in g["mae"] if x[h] is not None], dtype=float)
            b = np.array([x[h] for x in g["baseline_mean"]], dtype=float)
            valid = np.array([x[h] is not None for x in g["ret"]])
            b = b[valid]
            if len(r) == 0:
                gs.horizons[h] = {"n": 0}
                continue
            excess = r - b
            lo, hi = _bootstrap_ci(excess) if len(r) >= 2 else (float("nan"), float("nan"))
            sig = bool(len(r) >= MIN_SAMPLES and lo > 0)
            gs.horizons[h] = {
                "n": int(len(r)),
                "win_rate": float((r > 0).mean()),
                "mean": float(r.mean()),
                "median": float(np.median(r)),
                "mae_mean": float(m.mean()) if len(m) else None,
                "baseline_mean": float(b.mean()),
                "excess_mean": float(excess.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "significant": sig,
            }
        # 结论以 20 日持有期为准（报告里展示的就是它），其余持有期只在统计页上标色
        h20 = gs.horizons.get(20, {})
        if h20.get("n", 0) < MIN_SAMPLES:
            gs.verdict = "样本不足"
        elif h20.get("significant"):
            gs.verdict = "显著优于基准"
        else:
            gs.verdict = "不显著"
        gs.significant_horizons = [h for h in HORIZONS if gs.horizons.get(h, {}).get("significant")]
        out.append(gs)
    return out


def run_universe(
    market: Market,
    load_df: Callable[[Market, str], pd.DataFrame],
    symbols: list[str] | None = None,
    params: Params | None = None,
    paths: Paths | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[GroupStats], list[EventSample], dict]:
    paths = paths or Paths()
    params = params or load_params(market, paths)
    uni = load_universe(paths)[market]
    symbols = symbols or uni["symbols"]
    samples: list[EventSample] = []
    meta = {"market": market, "symbols": [], "failed": [], "bars": 0, "started": time.time()}
    for k, sym in enumerate(symbols):
        if progress:
            progress(sym, k, len(symbols))
        try:
            df = load_df(market, sym)
        except Exception as e:  # noqa: BLE001
            meta["failed"].append({"symbol": sym, "error": str(e)})
            continue
        if df.empty:
            meta["failed"].append({"symbol": sym, "error": "没有数据"})
            continue
        res = detect(df, params)
        samples.extend(collect_samples(market, sym, df, res))
        meta["symbols"].append(sym)
        meta["bars"] += len(df)
    meta["elapsed"] = time.time() - meta["started"]
    stats = summarize(samples)
    return stats, samples, meta


def save_stats(path: Path, stats_by_market: dict[str, list[GroupStats]], meta_by_market: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "horizons": list(HORIZONS),
        "min_samples": MIN_SAMPLES,
        "markets": {
            m: {"stats": [s.to_dict() for s in stats_by_market[m]], "meta": meta_by_market.get(m, {})}
            for m in stats_by_market
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stats_lookup(stats_json: dict, market: str) -> dict[str, dict]:
    """group -> stats dict，供报告按事件查找。"""
    mk = (stats_json.get("markets") or {}).get(market) or {}
    return {s["group"]: s for s in mk.get("stats", [])}


def event_group_for_lookup(e: Event) -> str | None:
    keys = group_keys(e)
    return keys[0] if keys else None


def kind_zh(kind: str) -> str:
    return EVENT_INFO.get(kind, {}).get("zh", kind)
