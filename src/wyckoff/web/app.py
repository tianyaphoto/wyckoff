"""FastAPI 应用。页面用 Jinja2 服务端渲染，局部刷新用 HTMX。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..analysis import Analyzer
from ..config import MARKETS, Market, Paths, Watchlist, normalize_code, validate_code_format
from ..data import DataService, SymbolNotFound
from ..data.sources import SourceError
from ..report.render import ReportRenderer
from .tasks import Task, TaskRunner, TaskStore

TEMPLATES = Path(__file__).parent / "templates"


def create_app(paths: Paths | None = None, service: DataService | None = None) -> FastAPI:
    paths = paths or Paths()
    service = service or DataService(paths)
    analyzer = Analyzer(paths, service)
    watchlist = Watchlist(paths)
    store = TaskStore(paths.tasks_db)

    def work(t: Task, set_status):
        set_status("fetching", "正在拉取历史数据")
        service.update(t.market, t.code)
        set_status("analyzing", "正在识别威克夫结构")
        analyzer.invalidate(t.market, t.code)
        analyzer.analyze(t.market, t.code)

    runner = TaskRunner(store, work)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="威克夫量价分析", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.analyzer = analyzer
    app.state.service = service
    app.state.watchlist = watchlist
    app.state.store = store
    app.state.runner = runner

    renderer = ReportRenderer(analyzer, web=True, link=lambda m, c: f"/symbol/{m}/{c}")
    renderer.env.loader.searchpath.append(str(TEMPLATES))  # type: ignore[attr-defined]

    def page(template: str, **ctx) -> HTMLResponse:
        return HTMLResponse(renderer.render(template, **ctx))

    def check_symbol(market: str, code: str) -> tuple[Market, str]:
        if market not in MARKETS:
            raise HTTPException(404, "市场不存在")
        code = normalize_code(market, code)  # type: ignore[arg-type]
        if not watchlist.contains(market, code):  # type: ignore[arg-type]
            raise HTTPException(404, f"{code} 不在自选清单里")
        return market, code  # type: ignore[return-value]

    # ------------------------------------------------------------ 页面
    @app.get("/", response_class=HTMLResponse)
    def overview():
        analyses = analyzer.analyze_many(watchlist.items())
        return HTMLResponse(renderer.overview_page(analyses))

    @app.get("/symbol/{market}/{code:path}", response_class=HTMLResponse)
    def symbol(market: str, code: str):
        market, code = check_symbol(market, code)
        a = analyzer.analyze(market, code)
        return HTMLResponse(renderer.symbol_page(a))

    @app.get("/watchlist", response_class=HTMLResponse)
    def watchlist_page():
        return page("watchlist.html", items=watchlist.load(), active="watchlist")

    @app.get("/stats", response_class=HTMLResponse)
    def stats_page():
        return HTMLResponse(renderer.stats_page(active="stats"))

    # ------------------------------------------------------------ 动作
    @app.post("/watchlist/add", response_class=HTMLResponse)
    def watchlist_add(market: str = Form(...), code: str = Form(...)):
        error = None
        ok = None
        if market not in MARKETS:
            error = "市场只能是美股或加密货币"
        else:
            norm = normalize_code(market, code)  # type: ignore[arg-type]
            error = validate_code_format(market, norm)  # type: ignore[arg-type]
            if not error and watchlist.contains(market, norm):  # type: ignore[arg-type]
                error = f"{norm} 已经在自选清单里"
            if not error:
                try:
                    if not service.exists(market, norm):  # type: ignore[arg-type]
                        error = f"没有找到 {norm} 这个代码"
                except SymbolNotFound:
                    error = f"没有找到 {norm} 这个代码"
                except SourceError as e:
                    error = f"数据源暂时不可用：{e}"
            if not error:
                watchlist.add(market, norm)  # type: ignore[arg-type]
                runner.submit("add", market, norm)
                ok = f"已添加 {norm}，正在拉取数据并分析"
        return page("_watchlist_panel.html", items=watchlist.load(), error=error, ok=ok, code_value="" if ok else code)

    @app.post("/watchlist/remove", response_class=HTMLResponse)
    def watchlist_remove(market: str = Form(...), code: str = Form(...)):
        try:
            watchlist.remove(market, code)  # type: ignore[arg-type]
            analyzer.invalidate(market, code)  # type: ignore[arg-type]
            ok = f"已从自选清单删除 {normalize_code(market, code)}，缓存数据保留"  # type: ignore[arg-type]
            error = None
        except ValueError as e:
            ok, error = None, str(e)
        return page("_watchlist_panel.html", items=watchlist.load(), error=error, ok=ok)

    @app.post("/symbol/{market}/{code:path}/delete")
    def symbol_delete(market: str, code: str):
        market, code = check_symbol(market, code)
        watchlist.remove(market, code)
        analyzer.invalidate(market, code)
        return RedirectResponse("/", status_code=303)

    @app.post("/symbol/{market}/{code:path}/reanalyze", response_class=HTMLResponse)
    def symbol_reanalyze(market: str, code: str):
        market, code = check_symbol(market, code)
        if not store.has_active(market, code):
            runner.submit("update", market, code)
        return page("_tasks.html", tasks=store.active_or_recent((market, code)))

    @app.post("/update-all", response_class=HTMLResponse)
    def update_all():
        for m, c in watchlist.items():
            if not store.has_active(m, c):
                runner.submit("update", m, c)
        return page("_tasks.html", tasks=store.active_or_recent())

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks(symbol: str | None = None):
        sym = None
        if symbol and ":" in symbol:
            m, c = symbol.split(":", 1)
            sym = (m, c)
        return page("_tasks.html", tasks=store.active_or_recent(sym))

    # ------------------------------------------------------------ JSON（测试和脚本用）
    @app.get("/api/watchlist")
    def api_watchlist():
        return watchlist.load()

    @app.get("/api/tasks")
    def api_tasks(symbol: str | None = None):
        sym = tuple(symbol.split(":", 1)) if symbol and ":" in symbol else None
        return [t.to_dict() for t in store.recent(limit=50, symbol=sym)]  # type: ignore[arg-type]

    @app.get("/api/tasks/{task_id}")
    def api_task(task_id: str):
        t = store.get(task_id)
        if not t:
            raise HTTPException(404, "任务不存在")
        return t.to_dict()

    @app.get("/api/overview")
    def api_overview():
        return [a.to_overview_row() for a in analyzer.analyze_many(watchlist.items())]

    @app.get("/api/symbol/{market}/{code:path}")
    def api_symbol(market: str, code: str):
        market, code = check_symbol(market, code)
        a = analyzer.analyze(market, code)
        return {
            "overview": a.to_overview_row(),
            "headline": a.headline,
            "events": [e.to_dict() for e in a.result.events],
            "structures": [s.to_dict() for s in a.result.structures],
        }

    @app.get("/health")
    def health():
        return JSONResponse({"ok": True})

    return app
