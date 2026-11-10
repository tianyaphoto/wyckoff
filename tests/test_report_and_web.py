"""报告、Web 服务、性能上限。全部用合成数据源，不联网。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from wyckoff.analysis import Analyzer
from wyckoff.data import DataService
from wyckoff.data.fake import FakeSource
from wyckoff.report import render_reports
from wyckoff.web import create_app


@pytest.fixture
def service(tmp_paths):
    return DataService(tmp_paths, sources={"us": FakeSource("us", 1200), "crypto": FakeSource("crypto", 1200)})


@pytest.fixture
def client(tmp_paths, service):
    app = create_app(tmp_paths, service)
    with TestClient(app) as c:
        yield c


def wait_task(client, task_id, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = client.get(f"/api/tasks/{task_id}").json()
        if not t["active"]:
            return t
        time.sleep(0.1)
    raise AssertionError("任务超时")


def latest_task(client, market, code):
    tasks = client.get("/api/tasks", params={"symbol": f"{market}:{code}"}).json()
    assert tasks, "没有任务"
    return tasks[0]


# ---------------------------------------------------------------- 报告


def test_render_reports_writes_all_files_in_chinese(tmp_paths, service):
    an = Analyzer(tmp_paths, service)
    items = [("us", "AAA"), ("us", "BBB"), ("crypto", "CCC/USDT")]
    written = render_reports(an, items)
    names = {p.name for p in written}
    assert names == {"us_AAA.html", "us_BBB.html", "crypto_CCC_USDT.html", "index.html", "stats.html", "guide.html"}
    html = (tmp_paths.reports / "us_AAA.html").read_text(encoding="utf-8")
    assert "Spring（弹簧/假跌破）" in html and "SC（抛售高潮）" in html
    assert "不构成投资建议" in html
    assert 'lang="zh-CN"' in html
    assert "Plotly.newPlot" in html
    idx = (tmp_paths.reports / "index.html").read_text(encoding="utf-8")
    assert "AAA" in idx and "CCC/USDT" in idx and "./us_AAA.html" in idx
    # 解释性内容：详情页的大白话解读、总览的读表说明、新手说明页
    assert "这是什么意思" in html and "接下来看什么" in html and "什么情况下作废" in html
    assert "这张表怎么看" in idx and "一句话解读" in idx
    guide = (tmp_paths.reports / "guide.html").read_text(encoding="utf-8")
    assert "新手说明" in guide and guide.count("<svg") == 2 and "长基线" in guide


def test_explain_covers_every_phase_and_no_structure(tmp_paths, service):
    from wyckoff.report.explain import explain, row_plain

    an = Analyzer(tmp_paths, service)
    a = an.analyze("us", "AAA")
    for kind in ("accumulation", "distribution", None):
        for phase in ("A", "B", "C", "D", "E") if kind else (None,):
            for cand in (None, "SC") if phase == "A" else (None,):
                a.current = {"kind": kind, "phase": phase, "confidence": "low", "trend": "震荡",
                             "lower": 90.0, "upper": 110.0, "candidate": cand}
                ex = explain(a)
                assert all(ex[k] and "None" not in ex[k] for k in ("title", "now", "watch", "invalid"))
                assert row_plain(a.current)


def test_guide_page_served(client):
    r = client.get("/guide")
    assert r.status_code == 200 and "新手说明" in r.text and "吸筹的五步" in r.text


def test_report_shows_only_confirmed_events(tmp_paths, service):
    an = Analyzer(tmp_paths, service)
    a = an.analyze("us", "AAA")
    last = a.df.index[-1]
    for ev in a.recent_events:
        assert ev.event.confirmed <= last


# ---------------------------------------------------------------- Web


def test_pages_render(client):
    for url in ("/", "/watchlist", "/stats", "/symbol/us/AAA", "/symbol/crypto/CCC/USDT"):
        r = client.get(url)
        assert r.status_code == 200, url
        assert "威克夫量价分析" in r.text
    assert client.get("/health").json() == {"ok": True}
    assert client.get("/symbol/us/NOTINLIST").status_code == 404


def test_add_valid_symbol_runs_task_and_appears_in_overview(client, tmp_paths):
    r = client.post("/watchlist/add", data={"market": "us", "code": "amd"})
    assert r.status_code == 200 and "已添加 AMD" in r.text
    assert "AMD" in tmp_paths.watchlist.read_text(encoding="utf-8")
    t = wait_task(client, latest_task(client, "us", "AMD")["id"])
    assert t["status"] == "done", t
    rows = client.get("/api/overview").json()
    assert any(row["code"] == "AMD" for row in rows)
    assert "AMD" in client.get("/").text
    assert (tmp_paths.data / "us" / "AMD.parquet").exists()


def test_add_unknown_code_is_rejected_and_config_untouched(client, tmp_paths):
    before = tmp_paths.watchlist.read_text(encoding="utf-8")
    r = client.post("/watchlist/add", data={"market": "us", "code": "NOPE"})
    assert "没有找到 NOPE" in r.text
    assert tmp_paths.watchlist.read_text(encoding="utf-8") == before
    r = client.post("/watchlist/add", data={"market": "us", "code": "bad code"})
    assert "只能是字母" in r.text


def test_add_duplicate_is_rejected(client):
    r = client.post("/watchlist/add", data={"market": "us", "code": "AAA"})
    assert "已经在自选清单里" in r.text
    assert client.get("/api/watchlist").json()["us"].count("AAA") == 1


def test_remove_symbol_keeps_cache(client, tmp_paths):
    client.get("/symbol/us/AAA")  # 触发缓存
    assert (tmp_paths.data / "us" / "AAA.parquet").exists()
    r = client.post("/watchlist/remove", data={"market": "us", "code": "AAA"})
    assert "已从自选清单删除 AAA" in r.text
    assert "AAA" not in client.get("/api/watchlist").json()["us"]
    assert "AAA" not in tmp_paths.watchlist.read_text(encoding="utf-8")
    assert (tmp_paths.data / "us" / "AAA.parquet").exists()
    assert client.get("/symbol/us/AAA").status_code == 404
    # 详情页上的删除按钮走另一条路由
    r = client.post("/symbol/us/BBB/delete", follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/watchlist").json()["us"] == []


def test_source_error_marks_task_failed_and_page_survives(client):
    r = client.post("/watchlist/add", data={"market": "us", "code": "BROKEN"})
    assert "已添加 BROKEN" in r.text
    t = wait_task(client, latest_task(client, "us", "BROKEN")["id"])
    assert t["status"] == "failed" and "超时" in t["message"]
    page = client.get("/")
    assert page.status_code == 200 and "BROKEN" in page.text
    tasks_html = client.get("/tasks").text
    assert "失败" in tasks_html and "重试" in tasks_html
    # 重试会再建一个任务
    r = client.post("/symbol/us/BROKEN/reanalyze")
    assert r.status_code == 200


def test_web_detail_matches_static_report(client, tmp_paths, service):
    api = client.get("/api/symbol/us/AAA").json()
    an = Analyzer(tmp_paths, service)
    a = an.analyze("us", "AAA")
    assert api["headline"] == a.headline
    assert [e["kind"] for e in api["events"]] == [e.kind for e in a.result.events]
    assert [s["lower"] for s in api["structures"]] == [s.lower for s in a.result.structures]
    static = render_reports(an, [("us", "AAA")], out_dir=tmp_paths.reports)
    html = (tmp_paths.reports / "us_AAA.html").read_text(encoding="utf-8")
    web = client.get("/symbol/us/AAA").text
    assert a.headline in html and a.headline in web


def test_update_all_enqueues_each_symbol(client):
    r = client.post("/update-all")
    assert r.status_code == 200
    tasks = client.get("/api/tasks").json()
    assert {t["code"] for t in tasks} >= {"AAA", "BBB", "CCC/USDT"}
    for t in tasks:
        wait_task(client, t["id"])


def test_serve_defaults_to_loopback():
    from wyckoff.cli import serve
    import inspect

    sig = inspect.signature(serve)
    assert sig.parameters["host"].default.default == "127.0.0.1"


# ---------------------------------------------------------------- 性能上限


def test_single_symbol_analysis_under_5s(tmp_paths):
    svc = DataService(tmp_paths, sources={"us": FakeSource("us", 5500), "crypto": FakeSource("crypto")})
    an = Analyzer(tmp_paths, svc)
    svc.update("us", "AAA")
    t0 = time.time()
    an.analyze("us", "AAA")
    assert time.time() - t0 < 5.0


def test_overview_page_under_1s_with_twelve_symbols(tmp_paths):
    svc = DataService(tmp_paths, sources={"us": FakeSource("us", 2500), "crypto": FakeSource("crypto", 2500)})
    codes = [f"S{k}" for k in range(12)]
    tmp_paths.watchlist.write_text("us: [" + ", ".join(codes) + "]\ncrypto: []\n", encoding="utf-8")
    for c in codes:
        svc.update("us", c)
    app = create_app(tmp_paths, svc)
    with TestClient(app) as client:
        client.get("/")  # 预热：第一次会做识别并缓存
        t0 = time.time()
        assert client.get("/").status_code == 200
        assert time.time() - t0 < 1.0


def test_add_symbol_appears_within_30s(client):
    t0 = time.time()
    client.post("/watchlist/add", data={"market": "crypto", "code": "DOGE"})
    wait_task(client, latest_task(client, "crypto", "DOGE/USDT")["id"], timeout=30)
    assert any(r["code"] == "DOGE/USDT" for r in client.get("/api/overview").json())
    assert time.time() - t0 < 30
