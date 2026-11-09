"""配置：项目路径、规则参数、自选清单、回测标的池。"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Market = Literal["us", "crypto"]
MARKETS: tuple[Market, ...] = ("us", "crypto")
MARKET_NAMES = {"us": "美股", "crypto": "加密货币"}


def project_root() -> Path:
    env = os.environ.get("WYCKOFF_HOME")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    root: Path = field(default_factory=project_root)

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def watchlist(self) -> Path:
        return self.config / "watchlist.yaml"

    @property
    def universe(self) -> Path:
        return self.config / "universe.yaml"

    @property
    def params(self) -> Path:
        return self.config / "params.yaml"

    @property
    def stats(self) -> Path:
        return self.data / "stats.json"

    @property
    def tasks_db(self) -> Path:
        return self.data / "tasks.db"


@dataclass(frozen=True)
class Params:
    atr_period: int = 14
    volume_period: int = 20
    trend_sma: int = 50
    extreme_lookback: int = 20
    warmup_bars: int = 50
    climax_range_atr: float = 2.0
    climax_volume_ratio: float = 2.5
    ar_window: int = 10
    ar_move_atr: float = 2.0
    ar_confirm_atr: float = 1.0
    ar_max_bars: int = 30
    range_min_bars: int = 15
    range_min_width_atr: float = 2.0
    range_max_width_atr: float = 8.0
    range_fail_atr: float = 1.5
    range_fail_bars: int = 3
    test_max_atr: float = 1.5
    test_recover_bars: int = 3
    test_low_volume_ratio: float = 1.0
    breakout_atr: float = 0.5
    breakout_volume_ratio: float = 1.5
    phase_d_bars: int = 20
    min_history_bars: int = 200


def load_params(market: Market, paths: Paths | None = None) -> Params:
    """读取 params.yaml，default 加上该市场的覆盖。"""
    paths = paths or Paths()
    raw: dict = {}
    if paths.params.exists():
        raw = yaml.safe_load(paths.params.read_text(encoding="utf-8")) or {}
    merged = dict(raw.get("default") or {})
    merged.update(raw.get(market) or {})
    valid = {k: v for k, v in merged.items() if k in Params.__dataclass_fields__}
    return Params(**valid)


# ---------------------------------------------------------------- 自选清单

_WATCHLIST_LOCK = threading.Lock()
_US_CODE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_CRYPTO_CODE = re.compile(r"^[A-Z0-9]{2,12}/USDT$")


def normalize_code(market: Market, code: str) -> str:
    code = code.strip().upper()
    if market == "crypto" and "/" not in code:
        code = f"{code}/USDT"
    return code


def validate_code_format(market: Market, code: str) -> str | None:
    """格式合法返回 None，否则返回中文错误说明。"""
    if market == "us":
        if not _US_CODE.match(code):
            return "美股代码只能是字母、数字、点和短横线，例如 NVDA 或 BRK-B"
    elif market == "crypto":
        if not _CRYPTO_CODE.match(code):
            return "加密货币只支持 USDT 交易对，例如 BTC/USDT"
    else:
        return "市场只能是 us 或 crypto"
    return None


class Watchlist:
    """自选清单，持久化在 watchlist.yaml。"""

    def __init__(self, paths: Paths | None = None):
        self.paths = paths or Paths()

    def load(self) -> dict[Market, list[str]]:
        if not self.paths.watchlist.exists():
            return {"us": [], "crypto": []}
        raw = yaml.safe_load(self.paths.watchlist.read_text(encoding="utf-8")) or {}
        return {m: [normalize_code(m, c) for c in (raw.get(m) or [])] for m in MARKETS}

    def save(self, data: dict[Market, list[str]]) -> None:
        self.paths.config.mkdir(parents=True, exist_ok=True)
        header = "# 自选清单。Web 页面和命令行都读写这个文件，它是自选清单的唯一来源。\n"
        body = yaml.safe_dump({m: list(data.get(m, [])) for m in MARKETS}, allow_unicode=True, sort_keys=False)
        tmp = self.paths.watchlist.with_suffix(".tmp")
        tmp.write_text(header + body, encoding="utf-8")
        tmp.replace(self.paths.watchlist)

    def items(self) -> list[tuple[Market, str]]:
        data = self.load()
        return [(m, c) for m in MARKETS for c in data[m]]

    def contains(self, market: Market, code: str) -> bool:
        return normalize_code(market, code) in self.load()[market]

    def add(self, market: Market, code: str) -> str:
        code = normalize_code(market, code)
        with _WATCHLIST_LOCK:
            data = self.load()
            if code in data[market]:
                raise ValueError(f"{code} 已经在自选清单里")
            data[market].append(code)
            self.save(data)
        return code

    def remove(self, market: Market, code: str) -> None:
        code = normalize_code(market, code)
        with _WATCHLIST_LOCK:
            data = self.load()
            if code not in data[market]:
                raise ValueError(f"{code} 不在自选清单里")
            data[market].remove(code)
            self.save(data)


# ---------------------------------------------------------------- 回测标的池


def load_universe(paths: Paths | None = None) -> dict[Market, dict]:
    paths = paths or Paths()
    raw = yaml.safe_load(paths.universe.read_text(encoding="utf-8")) or {}
    out: dict[Market, dict] = {}
    for m in MARKETS:
        sec = raw.get(m) or {}
        out[m] = {
            "start": sec.get("start"),
            "symbols": [normalize_code(m, c) for c in (sec.get("symbols") or [])],
        }
    return out
