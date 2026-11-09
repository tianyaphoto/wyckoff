"""第 2 层：证明识别和回测没有用到未来数据。任何一项不通过，统计结果作废。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wyckoff.backtest.stats import collect_samples
from wyckoff.detector import detect
from wyckoff.synthetic import active_market, random_walk


def _sig(events):
    return [(e.kind, e.occurred, e.confirmed, round(e.price, 6)) for e in events]


@pytest.fixture(scope="module")
def data():
    return active_market(2500, seed=7)


def test_truncation_keeps_confirmed_events_identical(data, params):
    full = detect(data, params)
    assert len(full.events) >= 20, "合成数据里应该有足够多的事件"
    rng = np.random.default_rng(0)
    cuts = sorted(set(rng.integers(100, len(data), size=120)))
    for t in cuts:
        part = detect(data.iloc[:t], params)
        cutoff = data.index[t - 1]
        expected = [e for e in full.events if e.confirmed <= cutoff]
        assert _sig(part.events) == _sig(expected), f"截断到第 {t} 天后事件不一致"
        # 阶段序列也必须逐日相同
        a = full.phases.iloc[:t][["structure_id", "kind", "phase", "confidence"]]
        b = part.phases[["structure_id", "kind", "phase", "confidence"]]
        pd.testing.assert_frame_equal(a, b)


def test_scrambling_the_future_changes_nothing_in_the_past(data, params):
    full = detect(data, params)
    rng = np.random.default_rng(1)
    for t in rng.integers(300, len(data) - 50, size=40):
        scrambled = data.copy()
        tail = data.iloc[t:].sample(frac=1, random_state=int(t)).to_numpy()
        scrambled.iloc[t:] = tail
        part = detect(scrambled, params)
        cutoff = data.index[t - 1]
        assert _sig([e for e in part.events if e.confirmed <= cutoff]) == _sig(
            [e for e in full.events if e.confirmed <= cutoff]
        )


def test_confirmation_never_precedes_occurrence(data, params):
    for e in detect(data, params).events:
        assert e.confirmed >= e.occurred


def test_backtest_entry_is_confirmation_close(data, params):
    res = detect(data, params)
    samples = collect_samples("us", "TEST", data, res)
    assert samples
    for s in samples:
        assert s.entry == pytest.approx(float(data.loc[pd.Timestamp(s.confirmed), "close"]))
        assert pd.Timestamp(s.confirmed) >= pd.Timestamp(s.occurred)


def test_forward_return_uses_only_bars_after_entry(data, params):
    res = detect(data, params)
    samples = collect_samples("us", "TEST", data, res)
    close = data["close"]
    for s in samples[:50]:
        i = data.index.get_loc(pd.Timestamp(s.confirmed))
        for h, r in s.ret.items():
            if r is None:
                assert i + h >= len(data)
            else:
                assert r == pytest.approx(s.side * (close.iloc[i + h] / close.iloc[i] - 1))
