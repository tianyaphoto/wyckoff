from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from wyckoff.config import Params, Paths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def params() -> Params:
    return Params()


@pytest.fixture
def tmp_paths(tmp_path: Path) -> Paths:
    """独立的项目目录：复制配置文件，数据和报告都写到临时目录。"""
    (tmp_path / "config").mkdir()
    for name in ("params.yaml", "universe.yaml"):
        shutil.copy(ROOT / "config" / name, tmp_path / "config" / name)
    (tmp_path / "config" / "watchlist.yaml").write_text("us: [AAA, BBB]\ncrypto: [CCC/USDT]\n", encoding="utf-8")
    return Paths(root=tmp_path)


def real_cache(market: str, code: str):
    """有真实缓存就返回数据，没有就跳过测试。"""
    from wyckoff.data import DataService

    svc = DataService(Paths(root=ROOT))
    if not svc.has_cache(market, code):
        pytest.skip(f"没有 {code} 的本地缓存，先运行 wyckoff update")
    return svc.read(market, code)
