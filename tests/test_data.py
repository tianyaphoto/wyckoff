"""数据层：缓存、增量更新、数据检查、复权。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from wyckoff.config import Paths, Watchlist, load_params, normalize_code, validate_code_format
from wyckoff.data import DataService, SymbolNotFound, check_quality
from wyckoff.data.fake import FakeSource
from wyckoff.synthetic import random_walk

from conftest import real_cache


def test_incremental_updates_equal_full_download(tmp_paths):
    src = FakeSource("us", bars=600)
    full = src._full("AAA")
    svc = DataService(tmp_paths, sources={"us": src, "crypto": FakeSource("crypto")})
    # 分三段模拟时间推进
    days = full.index
    for cut in (days[199].date(), days[399].date(), None):
        src.today = cut
        svc.update("us", "AAA")
    got = svc.read("us", "AAA")
    pd.testing.assert_frame_equal(got, full, check_freq=False)
    # 增量调用只请求最后几天之后的数据
    assert src.calls[-1][1] is not None and src.calls[-1][1] >= days[390].date()


def test_cache_roundtrip_and_delete(tmp_paths):
    svc = DataService(tmp_paths, sources={"us": FakeSource("us", 300), "crypto": FakeSource("crypto")})
    df = svc.update("us", "AAA")
    assert svc.has_cache("us", "AAA")
    pd.testing.assert_frame_equal(svc.read("us", "AAA"), df, check_freq=False)
    assert svc.last_updated("us", "AAA") == df.index[-1].date().isoformat()
    svc.delete_cache("us", "AAA")
    assert not svc.has_cache("us", "AAA")


def test_unknown_symbol_raises(tmp_paths):
    svc = DataService(tmp_paths, sources={"us": FakeSource("us"), "crypto": FakeSource("crypto")})
    with pytest.raises(SymbolNotFound):
        svc.update("us", "NOPE")
    assert not svc.exists("us", "NOPE")
    assert svc.exists("us", "AAA")


def test_quality_flags_short_history_and_jumps():
    df = random_walk(120, seed=0)
    q = check_quality(df, min_bars=200)
    assert any("少于 200" in w for w in q.warnings)
    df2 = random_walk(400, seed=0)
    df2.iloc[200:, df2.columns.get_loc("close")] *= 0.1  # 模拟未复权的 1 拆 10
    df2.iloc[200:, df2.columns.get_loc("open")] *= 0.1
    df2.iloc[200:, df2.columns.get_loc("high")] *= 0.1
    df2.iloc[200:, df2.columns.get_loc("low")] *= 0.1
    q2 = check_quality(df2, min_bars=200)
    assert q2.suspicious_jumps == 1
    assert check_quality(random_walk(400, seed=0), 200).ok


def test_watchlist_add_remove_and_validation(tmp_paths):
    wl = Watchlist(tmp_paths)
    assert wl.load() == {"us": ["AAA", "BBB"], "crypto": ["CCC/USDT"]}
    assert wl.add("crypto", "doge") == "DOGE/USDT"
    assert wl.contains("crypto", "DOGE")
    with pytest.raises(ValueError):
        wl.add("us", "aaa")
    wl.remove("us", "AAA")
    assert wl.load()["us"] == ["BBB"]
    text = tmp_paths.watchlist.read_text(encoding="utf-8")
    assert "DOGE/USDT" in text and "AAA" not in text
    assert validate_code_format("us", "BRK-B") is None
    assert validate_code_format("us", "bad code") is not None
    assert validate_code_format("crypto", "BTC/USDT") is None
    assert validate_code_format("crypto", "BTC/BUSD") is not None
    assert normalize_code("crypto", "eth") == "ETH/USDT"


def test_params_market_override(tmp_paths):
    (tmp_paths.config / "params.yaml").write_text("default:\n  ar_window: 10\ncrypto:\n  ar_window: 12\n", encoding="utf-8")
    assert load_params("us", tmp_paths).ar_window == 10
    assert load_params("crypto", tmp_paths).ar_window == 12
    assert load_params("crypto", tmp_paths).atr_period == 14  # 未覆盖的用默认


@pytest.mark.network
def test_nvda_split_is_adjusted():
    """NVDA 2024-06-10 1 拆 10：复权数据不应有跳空，也不应被识别成事件。"""
    df = real_cache("us", "NVDA")
    window = df.loc["2024-06-03":"2024-06-17"]
    assert window["close"].pct_change().abs().max() < 0.2
    from wyckoff.detector import detect

    r = detect(df, load_params("us"))
    split_day = pd.Timestamp("2024-06-10")
    assert not any(abs((e.occurred - split_day).days) <= 1 and e.kind in ("SC", "BC") for e in r.events)
    assert check_quality(df).suspicious_jumps == 0
