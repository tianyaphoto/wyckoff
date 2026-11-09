"""端到端：真实浏览器打开本地 Web 服务，新增 → 等待完成 → 打开详情 → 删除。每步截图存档。"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "reports" / "e2e"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    (home / "config").mkdir()
    for name in ("params.yaml", "universe.yaml"):
        shutil.copy(ROOT / "config" / name, home / "config" / name)
    (home / "config" / "watchlist.yaml").write_text("us: [AAA]\ncrypto: [BTC/USDT]\n", encoding="utf-8")
    port = _free_port()
    env = {**os.environ, "WYCKOFF_HOME": str(home), "WYCKOFF_FAKE_DATA": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "wyckoff.web.app:create_app", "--factory", "--host", "127.0.0.1", "--port", str(port)],
        env=env, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("服务没有启动")
    yield base, home
    proc.terminate()
    proc.wait(timeout=10)


def test_full_flow_in_browser(server):
    pw = pytest.importorskip("playwright.sync_api")
    base, home = server
    SHOTS.mkdir(parents=True, exist_ok=True)
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"没有安装 Chromium：{e}")
        page = browser.new_page(viewport={"width": 1366, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

        page.goto(base + "/")
        page.wait_for_selector("#overview-rows")
        page.screenshot(path=str(SHOTS / "01_overview.png"), full_page=True)

        page.click("text=管理自选清单")
        page.wait_for_selector("#code")
        page.screenshot(path=str(SHOTS / "02_watchlist.png"), full_page=True)

        page.select_option("#market", "us")
        page.fill("#code", "AMD")
        page.click("#add-btn")
        page.wait_for_selector("text=已添加 AMD", timeout=10000)
        page.screenshot(path=str(SHOTS / "03_added.png"), full_page=True)

        page.wait_for_selector(".task.done:has-text('AMD')", timeout=30000)
        page.screenshot(path=str(SHOTS / "04_task_done.png"), full_page=True)

        page.click("a.row-link:has-text('AMD')")
        page.wait_for_selector("#chart .plotly", timeout=15000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(SHOTS / "05_detail.png"), full_page=True)
        assert "AMD" in page.title()

        page.once("dialog", lambda d: d.accept())
        page.click("button:has-text('从自选清单删除')")
        page.wait_for_url(base + "/")
        page.wait_for_selector("#overview-rows")
        page.screenshot(path=str(SHOTS / "06_deleted.png"), full_page=True)
        assert "AMD" not in page.inner_text("#overview-rows")
        assert "AMD" not in (home / "config" / "watchlist.yaml").read_text(encoding="utf-8")

        # 事件统计页
        page.click("text=事件统计")
        page.wait_for_selector("h1:has-text('事件统计')")
        page.screenshot(path=str(SHOTS / "07_stats.png"), full_page=True)

        browser.close()
        bad = [e for e in errors if "favicon" not in e]
        assert not bad, f"浏览器报错：{bad}"
