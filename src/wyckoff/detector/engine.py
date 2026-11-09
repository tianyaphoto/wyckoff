"""逐日推进的状态机。第 i 天的所有判断只用第 0..i 天的数据。

吸筹和派发用同一套代码：把价格乘以方向 d（吸筹 +1，派发 -1）之后，派发结构就变成了吸筹结构。
所以镜像对称是由构造保证的，不依赖两份规则写得一样。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Params
from ..indicators import add_indicators
from .events import Event, Structure

IDLE, CLIMAX, RANGE, BREAKOUT = "idle", "climax", "range", "breakout"

# 方向 -> 事件名映射（吸筹 / 派发）
KIND_BY_DIR = {
    +1: {"climax": "SC", "test": "SPRING", "breakout": "SOS", "structure": "accumulation"},
    -1: {"climax": "BC", "test": "UTAD", "breakout": "SOW", "structure": "distribution"},
}


@dataclass
class DetectionResult:
    events: list[Event]
    structures: list[Structure]
    phases: pd.DataFrame  # index=date, columns: structure_id, kind, phase, confidence, trend, lower, upper
    frame: pd.DataFrame  # 带指标的原始数据

    def events_confirmed_by(self, date: pd.Timestamp) -> list[Event]:
        return [e for e in self.events if e.confirmed <= date]

    def current(self) -> dict:
        """最后一天的状态，供报告使用。"""
        if self.phases.empty:
            return {"phase": None, "confidence": None, "kind": None, "trend": None}
        row = self.phases.iloc[-1]
        latest = self.events[-1] if self.events else None
        return {
            "date": self.phases.index[-1],
            "kind": row["kind"] if isinstance(row["kind"], str) else None,
            "phase": row["phase"] if isinstance(row["phase"], str) else None,
            "confidence": row["confidence"] if isinstance(row["confidence"], str) else None,
            "trend": row["trend"],
            "lower": None if pd.isna(row["lower"]) else float(row["lower"]),
            "upper": None if pd.isna(row["upper"]) else float(row["upper"]),
            "candidate": row["candidate"] if isinstance(row["candidate"], str) else None,
            "latest_event": latest,
            "days_since_event": None if latest is None else int((self.phases.index[-1] - latest.confirmed).days),
        }


@dataclass
class _State:
    mode: str = IDLE
    d: int = 0
    struct: Structure | None = None
    # climax 阶段
    c_idx: int = -1
    c_low: float = 0.0  # 定向后的高潮极值
    h_max: float = -np.inf  # 定向后反弹最高
    h_idx: int = -1
    rebound: bool = False
    # range 阶段
    lower_o: float = 0.0
    upper_o: float = 0.0
    ar_idx: int = -1
    range_confirmed: bool = False
    fail_count: int = 0
    test_pending: bool = False
    test_idx: int = -1
    test_low: float = np.inf
    test_vol: float = np.nan
    away_from_lower: bool = True
    tests_confirmed: int = 0
    # breakout 阶段
    b_idx: int = -1
    extra: dict = field(default_factory=dict)


class Detector:
    def __init__(self, params: Params):
        self.p = params

    # ------------------------------------------------------------ 公共入口
    def run(self, df: pd.DataFrame) -> DetectionResult:
        p = self.p
        f = add_indicators(df, p)
        n = len(f)
        close = f["close"].to_numpy()
        high = f["high"].to_numpy()
        low = f["low"].to_numpy()
        atr = f["atr"].to_numpy()
        volr = f["vol_ratio"].to_numpy()
        sma = f["sma_trend"].to_numpy()
        low_n = f["low_n"].to_numpy()
        high_n = f["high_n"].to_numpy()
        rng = f["range"].to_numpy()
        dates = f.index

        events: list[Event] = []
        structures: list[Structure] = []
        st = _State()
        next_id = 1

        ph_sid = np.full(n, np.nan)
        ph_kind: list[str | None] = [None] * n
        ph_phase: list[str | None] = [None] * n
        ph_conf: list[str | None] = [None] * n
        ph_cand: list[str | None] = [None] * n
        ph_lower = np.full(n, np.nan)
        ph_upper = np.full(n, np.nan)
        ph_trend: list[str | None] = [None] * n

        def oriented(i: int, d: int) -> tuple[float, float, float]:
            """返回定向后的 (close, high, low)。派发时 high/low 互换并取负。"""
            if d == +1:
                return close[i], high[i], low[i]
            return -close[i], -low[i], -high[i]

        def climax_fires(i: int, d: int) -> bool:
            if d == +1:
                trend_ok = close[i] < sma[i]
                extreme_ok = low[i] < low_n[i]
            else:
                trend_ok = close[i] > sma[i]
                extreme_ok = high[i] > high_n[i]
            return bool(
                trend_ok and extreme_ok and rng[i] >= p.climax_range_atr * atr[i] and volr[i] >= p.climax_volume_ratio
            )

        def start_climax(i: int, d: int):
            nonlocal next_id, st
            _, _, ol = oriented(i, d)
            s = Structure(
                id=next_id,
                kind=KIND_BY_DIR[d]["structure"],
                climax_date=dates[i],
                climax_price=float(low[i] if d == +1 else high[i]),
            )
            next_id += 1
            st = _State(mode=CLIMAX, d=d, struct=s, c_idx=i, c_low=ol, h_max=-np.inf, h_idx=-1)

        def end_structure(i: int, reason: str):
            nonlocal st
            s = st.struct
            if s is not None:
                s.end = dates[i]
                s.end_reason = reason
                if s.ar_confirmed is not None:
                    structures.append(s)
            st = _State()

        def real_bounds(d: int, lower_o: float, upper_o: float) -> tuple[float, float]:
            a, b = d * lower_o, d * upper_o
            return (min(a, b), max(a, b))

        def emit(kind: str, i_occ: int, i_conf: int, price: float, rule: str, vol: float | None, vclass=None):
            s = st.struct
            e = Event(
                kind=kind,
                structure=s.kind,
                structure_id=s.id,
                occurred=dates[i_occ],
                confirmed=dates[i_conf],
                price=float(price),
                atr=float(atr[i_conf]),
                vol_ratio=None if vol is None or np.isnan(vol) else float(vol),
                rule=rule,
                volume_class=vclass,
            )
            events.append(e)
            s.events.append(e)
            return e

        for i in range(n):
            valid = (
                i >= p.warmup_bars
                and not np.isnan(atr[i])
                and not np.isnan(volr[i])
                and not np.isnan(sma[i])
                and not np.isnan(low_n[i])
                and atr[i] > 0
            )
            # 趋势标签：50 日均线最近 10 根的变化
            if valid and i >= 10 and not np.isnan(sma[i - 10]):
                slope = sma[i] - sma[i - 10]
                ph_trend[i] = "上升趋势" if slope > 0.5 * atr[i] else ("下降趋势" if slope < -0.5 * atr[i] else "震荡")
            if not valid:
                continue

            d = st.d
            if st.mode == IDLE:
                for dd in (+1, -1):
                    if climax_fires(i, dd):
                        start_climax(i, dd)
                        break

            elif st.mode == CLIMAX:
                oc, oh, ol = oriented(i, d)
                k = KIND_BY_DIR[d]
                bars = i - st.c_idx
                # 更深的新高潮：重新锚定
                if climax_fires(i, d) and ol < st.c_low:
                    start_climax(i, d)
                elif not st.rebound and ol < st.c_low and bars <= p.ar_window:
                    # 高潮延续：反弹出现前继续创新低，高潮极值顺延到最低那天
                    st.c_idx, st.c_low = i, ol
                    st.struct.climax_date = dates[i]
                    st.struct.climax_price = float(low[i] if d == +1 else high[i])
                    st.h_max, st.h_idx = -np.inf, -1
                else:
                    if oh > st.h_max:
                        st.h_max, st.h_idx = oh, i
                    if not st.rebound and bars <= p.ar_window and st.h_max - st.c_low >= p.ar_move_atr * atr[i]:
                        st.rebound = True
                    if not st.rebound and bars > p.ar_window:
                        end_structure(i, "abandoned")
                    elif bars > p.ar_max_bars:
                        end_structure(i, "abandoned")
                    elif st.rebound and oc <= st.h_max - p.ar_confirm_atr * atr[i] and st.h_idx < i:
                        # AR 确认
                        lower_o, upper_o = st.c_low, st.h_max
                        width = upper_o - lower_o
                        if not (p.range_min_width_atr * atr[i] <= width <= p.range_max_width_atr * atr[i]):
                            end_structure(i, "abandoned")
                        else:
                            s = st.struct
                            lo, up = real_bounds(d, lower_o, upper_o)
                            s.lower, s.upper, s.range_atr = lo, up, float(atr[i])
                            s.ar_date = dates[st.h_idx]
                            s.ar_price = float(d * st.h_max)
                            s.ar_confirmed = dates[i]
                            cdir = "下跌" if d == +1 else "上涨"
                            emit(
                                k["climax"],
                                st.c_idx,
                                i,
                                s.climax_price,
                                f"{cdir}趋势中（收盘在 {p.trend_sma} 日均线{'下' if d == 1 else '上'}方），"
                                f"当日振幅 ≥ {p.climax_range_atr:g} 倍 ATR，量比 ≥ {p.climax_volume_ratio:g}，"
                                f"创 {p.extreme_lookback} 日新{'低' if d == 1 else '高'}；在 AR 确认时一并确认",
                                volr[st.c_idx],
                            )
                            emit(
                                "AR",
                                st.h_idx,
                                i,
                                s.ar_price,
                                f"高潮后 {p.ar_window} 根 K 线内反向运动 ≥ {p.ar_move_atr:g} 倍 ATR 的极值，"
                                f"从极值回撤 ≥ {p.ar_confirm_atr:g} 倍 ATR 时确认",
                                volr[st.h_idx],
                            )
                            st.mode = RANGE
                            st.lower_o, st.upper_o, st.ar_idx = lower_o, upper_o, i
                            st.fail_count = 0
                            st.away_from_lower = True

            elif st.mode == RANGE:
                oc, oh, ol = oriented(i, d)
                k = KIND_BY_DIR[d]
                s = st.struct
                lower_o, upper_o = st.lower_o, st.upper_o
                bars_in_range = i - st.ar_idx
                # 区间失败：收盘超出边界太多且没有收回
                beyond = oc > upper_o + p.range_fail_atr * atr[i] or oc < lower_o - p.range_fail_atr * atr[i]
                inside = lower_o <= oc <= upper_o
                if beyond:
                    st.fail_count += 1
                elif inside:
                    st.fail_count = 0
                # 更深的新高潮：之前的高潮不是真正的高潮，重新锚定
                deeper_climax = climax_fires(i, d) and ol < lower_o - p.test_max_atr * atr[i]
                if deeper_climax:
                    end_structure(i, "new_climax")
                    start_climax(i, d)
                elif st.fail_count >= p.range_fail_bars:
                    end_structure(i, "failed")
                    # 失败当天也可能是新的高潮
                    for dd in (+1, -1):
                        if climax_fires(i, dd):
                            start_climax(i, dd)
                            break
                else:
                    if not st.range_confirmed and bars_in_range >= p.range_min_bars:
                        st.range_confirmed = True
                        s.range_confirmed = dates[i]
                        emit(
                            "RANGE",
                            st.c_idx,
                            i,
                            s.climax_price,
                            f"{k['climax']} 低点与 AR 高点围成区间，宽度 {p.range_min_width_atr:g}–{p.range_max_width_atr:g} 倍 ATR，"
                            f"AR 确认后价格停留 ≥ {p.range_min_bars} 根 K 线"
                            if d == 1
                            else f"{k['climax']} 高点与 AR 低点围成区间，宽度 {p.range_min_width_atr:g}–{p.range_max_width_atr:g} 倍 ATR，"
                            f"AR 确认后价格停留 ≥ {p.range_min_bars} 根 K 线",
                            None,
                        )
                    if st.range_confirmed:
                        # 突破
                        if oc > upper_o + p.breakout_atr * atr[i] and volr[i] >= p.breakout_volume_ratio:
                            emit(
                                k["breakout"],
                                i,
                                i,
                                close[i],
                                f"收盘{'高于上沿' if d == 1 else '低于下沿'} ≥ {p.breakout_atr:g} 倍 ATR，"
                                f"量比 ≥ {p.breakout_volume_ratio:g}",
                                volr[i],
                            )
                            s.breakout = dates[i]
                            st.mode = BREAKOUT
                            st.b_idx = i
                            st.test_pending = False
                        else:
                            # 测试（Spring / UTAD）
                            depth = lower_o - ol
                            if ol < lower_o and depth <= p.test_max_atr * atr[i] and st.away_from_lower:
                                if not st.test_pending or ol < st.test_low:
                                    if not st.test_pending:
                                        st.test_idx = i
                                    elif ol < st.test_low:
                                        st.test_idx = i
                                    st.test_low = ol
                                    st.test_vol = volr[i]
                                    st.test_pending = True
                            elif ol < lower_o - p.test_max_atr * atr[i]:
                                st.test_pending = False  # 刺得太深，不是测试
                            if st.test_pending:
                                if oc > lower_o:
                                    vclass = "缩量" if st.test_vol < p.test_low_volume_ratio else "放量"
                                    emit(
                                        k["test"],
                                        st.test_idx,
                                        i,
                                        d * st.test_low,
                                        f"最{'低' if d == 1 else '高'}价刺破{'下沿' if d == 1 else '上沿'}不超过 "
                                        f"{p.test_max_atr:g} 倍 ATR，{p.test_recover_bars} 根 K 线内收盘回到区间内；"
                                        f"刺破日量比 {st.test_vol:.2f}（{vclass}）",
                                        st.test_vol,
                                        vclass,
                                    )
                                    s.test_confirmed = dates[i]
                                    st.tests_confirmed += 1
                                    st.test_pending = False
                                    st.away_from_lower = False
                                elif i - st.test_idx >= p.test_recover_bars:
                                    st.test_pending = False
                            if not st.away_from_lower and oc >= lower_o + atr[i]:
                                st.away_from_lower = True

            elif st.mode == BREAKOUT:
                oc, oh, ol = oriented(i, d)
                s = st.struct
                # 新高潮（任一方向）优先
                started = False
                for dd in (+1, -1):
                    if climax_fires(i, dd):
                        end_structure(i, "new_climax")
                        start_climax(i, dd)
                        started = True
                        break
                if not started and oc < st.upper_o:
                    end_structure(i, "breakout_failed")

            # ---- 记录当天阶段
            s = st.struct
            if s is not None and st.mode != IDLE:
                ph_sid[i] = s.id
                ph_kind[i] = s.kind
                if st.mode == CLIMAX:
                    ph_phase[i] = "A"
                    ph_conf[i] = "low"
                    ph_cand[i] = KIND_BY_DIR[st.d]["climax"]
                elif st.mode == RANGE:
                    ph_phase[i] = "C" if st.tests_confirmed > 0 else "B"
                    ph_conf[i] = "high" if st.tests_confirmed > 0 else ("medium" if st.range_confirmed else "low")
                    ph_lower[i], ph_upper[i] = s.lower, s.upper
                elif st.mode == BREAKOUT:
                    ph_phase[i] = "D" if i - st.b_idx <= p.phase_d_bars else "E"
                    ph_conf[i] = "high" if st.tests_confirmed > 0 else "medium"
                    ph_lower[i], ph_upper[i] = s.lower, s.upper

        # 收尾：仍在进行中的结构
        if st.struct is not None and st.struct.ar_confirmed is not None:
            st.struct.end_reason = "ongoing"
            structures.append(st.struct)

        phases = pd.DataFrame(
            {
                "structure_id": ph_sid,
                "kind": ph_kind,
                "phase": ph_phase,
                "confidence": ph_conf,
                "candidate": ph_cand,
                "trend": ph_trend,
                "lower": ph_lower,
                "upper": ph_upper,
            },
            index=dates,
        )
        return DetectionResult(events=events, structures=structures, phases=phases, frame=f)


def detect(df: pd.DataFrame, params: Params) -> DetectionResult:
    return Detector(params).run(df)
