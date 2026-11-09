"""第 3 层：用已知答案的数据检验回测统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wyckoff.backtest.stats import HORIZONS, MIN_SAMPLES, collect_samples, run_universe, summarize
from wyckoff.detector import detect
from wyckoff.synthetic import active_market, random_walk


def _samples_for(n_symbols: int, seed0: int, params, **kw):
    out = []
    for k in range(n_symbols):
        df = active_market(2000, seed=seed0 + k, **kw)
        out.extend(collect_samples("us", f"S{k}", df, detect(df, params)))
    return out


@pytest.mark.slow
def test_placebo_random_walk_has_about_five_percent_false_positives(params):
    """随机游走里不存在真实优势，显著的分组比例应接近 5%。"""
    samples = _samples_for(40, 100, params)
    stats = summarize(samples)
    tested = [(s, h) for s in stats for h in HORIZONS if s.horizons[h]["n"] >= MIN_SAMPLES]
    assert len(tested) >= 10
    sig = sum(1 for s, h in tested if s.horizons[h]["significant"])
    rate = sig / len(tested)
    assert rate <= 0.20, f"安慰剂数据上 {rate:.0%} 的分组显著，统计方法可能有 bug"


def test_planted_signal_is_detected(params):
    """在 Spring 确认后 20 天人为植入 +5% 收益，统计必须抓到。"""
    samples = []
    for k in range(25):
        df = active_market(2000, seed=500 + k)
        res = detect(df, params)
        springs = [e for e in res.events if e.kind == "SPRING"]
        close = df["close"].to_numpy().copy()
        planted: set = set()
        last_i = -100
        for e in springs:
            i = df.index.get_loc(e.confirmed)
            # 相邻 Spring 间隔不足 25 根时跳过，避免一次植入影响另一个样本的收益
            if i + 20 < len(close) and i - last_i > 25:
                target = close[i] * 1.05
                df.iloc[i + 20, df.columns.get_loc("close")] = target
                df.iloc[i + 20, df.columns.get_loc("high")] = max(df.iloc[i + 20]["high"], target)
                df.iloc[i + 20, df.columns.get_loc("low")] = min(df.iloc[i + 20]["low"], target)
                planted.add(e.confirmed.date().isoformat())
                last_i = i
        # 植入后重新识别。改动都在确认日之后，所以这些 Spring 本身不变（截断测试保证）；
        # 但改动可能让后面多出或少掉别的事件，只统计植入过的样本。
        res2 = detect(df, params)
        samples.extend(
            [s for s in collect_samples("us", f"P{k}", df, res2) if s.group == "SPRING" and s.confirmed in planted]
        )
    stats = {s.group: s for s in summarize(samples)}
    h20 = stats["SPRING"].horizons[20]
    assert h20["n"] >= MIN_SAMPLES
    assert h20["mean"] == pytest.approx(0.05, abs=0.002)
    assert h20["win_rate"] == 1.0
    assert h20["significant"] is True
    assert stats["SPRING"].verdict == "显著优于基准"


def test_independent_recomputation_of_ten_events(params):
    df = active_market(2000, seed=9)
    res = detect(df, params)
    samples = collect_samples("us", "X", df, res)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), size=min(10, len(samples)), replace=False)
    close = df["close"]
    for j in pick:
        s = samples[j]
        i = df.index.get_loc(pd.Timestamp(s.confirmed))
        for h in HORIZONS:
            if i + h < len(df):
                manual = s.side * (close.iloc[i + h] / close.iloc[i] - 1)
                assert s.ret[h] == pytest.approx(manual, abs=1e-12)
                path = df.iloc[i + 1 : i + h + 1]
                manual_mae = (path["low"].min() / close.iloc[i] - 1) if s.side > 0 else -(path["high"].max() / close.iloc[i] - 1)
                assert s.mae[h] == pytest.approx(min(manual_mae, 0.0), abs=1e-12)


def test_baseline_is_any_day_entry_mean(params):
    df = active_market(1000, seed=2)
    samples = collect_samples("us", "X", df, detect(df, params))
    long = [s for s in samples if s.side > 0]
    if not long:
        pytest.skip("没有做多事件")
    c = df["close"].to_numpy()
    for h in HORIZONS:
        manual = np.mean(c[h:] / c[:-h] - 1)
        assert long[0].baseline_mean[h] == pytest.approx(manual)


def test_small_groups_get_no_verdict(params):
    df = active_market(400, seed=5)
    stats = summarize(collect_samples("us", "X", df, detect(df, params)))
    for s in stats:
        if s.horizons[20]["n"] < MIN_SAMPLES:
            assert s.verdict == "样本不足"


def test_run_universe_collects_failures_and_meta(params, tmp_paths):
    def load(market, sym):
        if sym == "BAD":
            raise RuntimeError("模拟失败")
        return active_market(800, seed=1)

    stats, samples, meta = run_universe("us", load, symbols=["A", "BAD", "B"], params=params, paths=tmp_paths)
    assert meta["symbols"] == ["A", "B"]
    assert meta["failed"][0]["symbol"] == "BAD"
    assert meta["bars"] == 1600
