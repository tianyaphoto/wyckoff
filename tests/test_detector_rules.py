"""第 1 层：规格逐条对照。每条规则都卡在阈值两边各造一份数据。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from wyckoff.config import Params
from wyckoff.detector import detect
from wyckoff.synthetic import ScenarioBuilder, accumulation_scenario, active_market, mirror, random_walk

MIRROR = {"SC": "BC", "BC": "SC", "SPRING": "UTAD", "UTAD": "SPRING", "SOS": "SOW", "SOW": "SOS", "AR": "AR", "RANGE": "RANGE"}


def kinds(result):
    return [e.kind for e in result.events]


def event(result, kind):
    return next(e for e in result.events if e.kind == kind)


# ---------------------------------------------------------------- 教科书场景


def test_textbook_accumulation_finds_all_eight_core_events(params):
    df, marks = accumulation_scenario()
    r = detect(df, params)
    assert kinds(r) == ["SC", "AR", "RANGE", "SPRING", "SOS"]
    assert df.index.get_loc(event(r, "SC").occurred) == marks["SC"]
    assert df.index.get_loc(event(r, "AR").occurred) == marks["AR"]
    assert df.index.get_loc(event(r, "SPRING").occurred) == marks["SPRING"]
    assert df.index.get_loc(event(r, "SOS").occurred) == marks["SOS"]
    assert event(r, "SC").price == pytest.approx(marks["sc_low"])
    assert event(r, "AR").price == pytest.approx(marks["ar_high"])
    s = r.structures[0]
    assert s.kind == "accumulation"
    assert s.lower == pytest.approx(marks["sc_low"]) and s.upper == pytest.approx(marks["ar_high"])
    assert s.end_reason == "ongoing" and s.breakout is not None


def test_textbook_distribution_is_exact_mirror(params):
    df, _ = accumulation_scenario()
    r = detect(df, params)
    rm = detect(mirror(df), params)
    assert [(MIRROR[e.kind], e.occurred, e.confirmed) for e in r.events] == [
        (e.kind, e.occurred, e.confirmed) for e in rm.events
    ]
    assert rm.structures[0].kind == "distribution"
    pivot = float(df["close"].iloc[0])
    for e, em in zip(r.events, rm.events):
        assert em.price == pytest.approx(2 * pivot - e.price)


# ---------------------------------------------------------------- SC 阈值


@pytest.mark.parametrize("ratio,expect", [(2.50, True), (2.49, False)])
def test_climax_volume_ratio_threshold(params, ratio, expect):
    # 高潮前 20 根成交量恰好都是 1e6，所以量比 = climax_volume / 1e6
    df, _ = accumulation_scenario(climax_volume=ratio * 1e6)
    r = detect(df, params)
    assert ("SC" in kinds(r)) is expect


def test_climax_range_threshold(params):
    small = Params(**{**params.__dict__, "climax_range_atr": 50.0})
    df, _ = accumulation_scenario()
    assert "SC" in kinds(detect(df, params))
    assert "SC" not in kinds(detect(df, small))


def test_climax_requires_downtrend(params):
    """同样的高潮 K 线放在上涨趋势里不是 SC。"""
    df, marks = accumulation_scenario()
    # 把高潮前 30 根的下跌改成上涨，收盘就在 50 日均线上方
    up = df.copy()
    seg = slice(60, marks["SC"])
    shift = np.linspace(0, 24, marks["SC"] - 60)  # 抵消 -0.4*30 的下跌并转为上涨
    for col in ("open", "high", "low", "close"):
        up.iloc[seg, up.columns.get_loc(col)] = df.iloc[seg][col].to_numpy() + shift
    # 高潮日本身也抬高，使其不再创 20 日新低但仍是大振幅放量 K 线
    r = detect(up, params)
    assert "SC" not in kinds(r) or event(r, "SC").occurred != df.index[marks["SC"]]


# ---------------------------------------------------------------- AR


def test_ar_must_arrive_within_window(params):
    strict = Params(**{**params.__dict__, "ar_window": 2})
    df, _ = accumulation_scenario()
    assert "AR" in kinds(detect(df, params))
    assert "AR" not in kinds(detect(df, strict))


def test_ar_confirmed_on_pullback_not_on_high(params):
    df, marks = accumulation_scenario()
    ar = event(detect(df, params), "AR")
    assert df.index.get_loc(ar.confirmed) > marks["AR"]
    assert df.index.get_loc(ar.confirmed) <= marks["AR_confirmed"]


# ---------------------------------------------------------------- 区间


@pytest.mark.parametrize("bars,expect", [(15, True), (14, False)])
def test_range_needs_min_bars_after_ar(params, bars, expect):
    full, marks = accumulation_scenario(range_bars=30, with_spring=False, with_sos=False)
    ar_idx = full.index.get_loc(event(detect(full, params), "AR").confirmed)
    df = full.iloc[: ar_idx + bars + 1]  # AR 确认后恰好 bars 根 K 线
    r = detect(df, params)
    assert ("RANGE" in kinds(r)) is expect
    if expect:
        rng = event(r, "RANGE")
        assert df.index.get_loc(rng.confirmed) - ar_idx == params.range_min_bars
        assert rng.occurred == event(r, "SC").occurred


def test_range_width_must_be_within_atr_bounds(params):
    df, _ = accumulation_scenario()
    r = detect(df, params)
    s = r.structures[0]
    width_atr = (s.upper - s.lower) / s.range_atr
    assert params.range_min_width_atr <= width_atr <= params.range_max_width_atr
    too_narrow = Params(**{**params.__dict__, "range_min_width_atr": width_atr + 0.01})
    too_wide = Params(**{**params.__dict__, "range_max_width_atr": width_atr - 0.01})
    assert detect(df, too_narrow).structures == []
    assert detect(df, too_wide).structures == []


def test_range_fails_when_close_leaves_without_recovery(params):
    df, marks = accumulation_scenario(with_spring=False, with_sos=False, range_bars=20)
    # 区间确认后连续 3 天收盘跌到下沿下方远超 1.5 ATR（此时 ATR 约 1.9）
    extra = []
    price = marks["sc_low"] - 4.5
    for _ in range(4):
        extra.append((price + 0.3, price + 0.5, price - 0.5, price, 1e6))
    idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=4)
    df2 = pd.concat([df, pd.DataFrame(extra, columns=df.columns, index=idx)])
    r = detect(df2, params)
    assert r.structures[0].end_reason == "failed"
    assert "SPRING" not in kinds(r)


# ---------------------------------------------------------------- Spring / SOS


@pytest.mark.parametrize("depth,expect", [(1.0, True), (3.0, False)])
def test_spring_depth_limit(params, depth, expect):
    df, _ = accumulation_scenario(spring_depth=depth, with_sos=False)
    r = detect(df, params)
    assert ("SPRING" in kinds(r)) is expect


@pytest.mark.parametrize("vol,cls", [(0.6e6, "缩量"), (1.5e6, "放量")])
def test_spring_volume_class(params, vol, cls):
    df, _ = accumulation_scenario(spring_volume=vol, with_sos=False)
    assert event(detect(df, params), "SPRING").volume_class == cls


def test_spring_requires_close_back_inside(params):
    df, marks = accumulation_scenario(with_spring=False, with_sos=False, range_bars=20)
    # 刺破后 4 天都收在下沿下方：不是 Spring
    lo = marks["sc_low"]
    extra = [(lo + 0.2, lo + 0.3, lo - 0.8, lo - 0.3, 0.6e6)] + [(lo - 0.3, lo - 0.1, lo - 0.9, lo - 0.4, 0.8e6)] * 4
    idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=len(extra))
    df2 = pd.concat([df, pd.DataFrame(extra, columns=df.columns, index=idx)])
    assert "SPRING" not in kinds(detect(df2, params))


def test_sos_needs_volume(params):
    df, _ = accumulation_scenario(sos_volume=1.2e6)  # 量比 ~1.3 < 1.5
    assert "SOS" not in kinds(detect(df, params))
    df, _ = accumulation_scenario(sos_volume=2e6)
    assert "SOS" in kinds(detect(df, params))


def test_events_before_range_confirmation_are_not_signals(params):
    """区间满 15 根之前不识别 Spring/SOS。"""
    df, marks = accumulation_scenario(range_bars=5, with_sos=False)
    r = detect(df, params)
    assert "RANGE" not in kinds(r)
    assert "SPRING" not in kinds(r)


# ---------------------------------------------------------------- 尺度和镜像（不变量）


def test_scaling_prices_tenfold_changes_nothing(params):
    df = active_market(1500, seed=3)
    r1 = detect(df, params)
    scaled = df.copy()
    for c in ("open", "high", "low", "close"):
        scaled[c] = df[c] * 10
    r2 = detect(scaled, params)
    assert len(r1.events) > 5
    assert [(e.kind, e.occurred, e.confirmed) for e in r1.events] == [(e.kind, e.occurred, e.confirmed) for e in r2.events]
    for a, b in zip(r1.events, r2.events):
        assert b.price == pytest.approx(a.price * 10)


def test_mirror_on_random_walk(params):
    df = active_market(1500, seed=4)
    r1 = detect(df, params)
    r2 = detect(mirror(df), params)
    assert len(r1.events) > 5
    assert [(MIRROR[e.kind], e.occurred, e.confirmed) for e in r1.events] == [
        (e.kind, e.occurred, e.confirmed) for e in r2.events
    ]


# ---------------------------------------------------------------- 阶段和可信度


def _phase_at_end(df, params):
    r = detect(df, params)
    row = r.phases.iloc[-1]
    return row["phase"], row["confidence"], r


def test_phase_progression(params):
    df, marks = accumulation_scenario(range_bars=20)
    p = params
    # A：高潮已出现，AR 未确认
    ph, conf, r = _phase_at_end(df.iloc[: marks["SC"] + 1], p)
    assert (ph, conf) == ("A", "low") and r.phases.iloc[-1]["candidate"] == "SC"
    # B 低：AR 已确认，区间未满 15 根
    ph, conf, _ = _phase_at_end(df.iloc[: marks["AR_confirmed"] + 1], p)
    assert (ph, conf) == ("B", "low")
    # B 中：区间已满 15 根
    ar_idx = df.index.get_loc(event(detect(df, p), "AR").confirmed)
    ph, conf, _ = _phase_at_end(df.iloc[: ar_idx + p.range_min_bars + 1], p)
    assert (ph, conf) == ("B", "medium")
    # C 高：Spring 确认
    ph, conf, _ = _phase_at_end(df.iloc[: marks["SPRING"] + 1], p)
    assert (ph, conf) == ("C", "high")
    # D：突破后 20 根内
    ph, conf, _ = _phase_at_end(df.iloc[: marks["SOS"] + 3], p)
    assert (ph, conf) == ("D", "high")
    # E：突破后超过 20 根（追加上涨 K 线）
    tail = df.iloc[-1]
    extra = []
    price = tail["close"]
    for _ in range(25):
        price += 0.3
        extra.append((price - 0.3, price + 0.4, price - 0.6, price, 1e6))
    idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=25)
    df2 = pd.concat([df, pd.DataFrame(extra, columns=df.columns, index=idx)])
    ph, conf, _ = _phase_at_end(df2, p)
    assert (ph, conf) == ("E", "high")


def test_no_structure_shows_trend_label(params):
    df = ScenarioBuilder().calm(300, drift=0.15).frame()
    r = detect(df, params)
    assert r.events == []
    cur = r.current()
    assert cur["phase"] is None
    assert cur["trend"] in ("上升趋势", "下降趋势", "震荡")


def test_pure_trend_without_consolidation_has_no_structure(params):
    """反面案例：单边强趋势、没有横盘，不应硬套出区间。"""
    b = ScenarioBuilder()
    b.calm(60)
    for k in range(300):  # 稳步上涨，偶尔放量但没有高潮式的大振幅
        o = b.price
        b.bar(o, o + 1.0, o - 0.3, o + 0.6, 1.8e6 if k % 7 == 0 else 1e6)
    r = detect(b.frame(), params)
    assert r.structures == []
    assert r.current()["trend"] == "上升趋势"


# ---------------------------------------------------------------- 长基线（慢崩盘）


def slow_crash_scenario() -> tuple[pd.DataFrame, int]:
    """波动和成交量先慢慢放大 25 根，再出现高潮日：短基线被抬高后认不出，长基线认得出。"""
    b = ScenarioBuilder(200.0)
    b.calm(80)
    for k in range(25):
        amp = 1.0 + 1.3 * k / 24
        vol = 1e6 * (1.0 + 1.2 * k / 24)
        o = b.price
        c = o - 0.8
        b.bar(o, o + amp / 2 - 0.4, c - amp / 2 + 0.4, c, vol)
    sc = len(b.rows)
    o = b.price
    b.bar(o, o + 0.2, o - 3.4, o - 2.8, 4.0e6)
    for _ in range(5):  # AR：反弹
        o = b.price
        b.bar(o, o + 1.8, o - 0.3, o + 1.4, 1.5e6)
    for _ in range(3):  # 回落确认
        o = b.price
        b.bar(o, o + 0.3, o - 1.6, o - 1.2, 1e6)
    b.calm(10)
    return b.frame(), sc


def test_long_baseline_catches_climax_after_slow_crash(params):
    df, sc = slow_crash_scenario()
    assert "SC" not in kinds(detect(df, params)), "短基线下这一天不该被认成 SC"
    r = detect(df, replace(params, long_baseline_bars=60))
    e = event(r, "SC")
    assert df.index.get_loc(e.occurred) == sc
    assert "60 日长基线" in e.rule


def test_long_baseline_off_by_default_and_adds_nothing_to_textbook_case(params):
    assert params.long_baseline_bars == 0
    df, _ = accumulation_scenario()
    a = detect(df, params)
    b = detect(df, replace(params, long_baseline_bars=60))
    assert [(e.kind, e.occurred) for e in a.events] == [(e.kind, e.occurred) for e in b.events]


def test_long_baseline_mirror_symmetry(params):
    df, _ = slow_crash_scenario()
    p = replace(params, long_baseline_bars=60)
    a = detect(df, p)
    m = detect(mirror(df, 400.0), p)
    assert [(MIRROR[e.kind], e.occurred) for e in a.events] == [(e.kind, e.occurred) for e in m.events]
