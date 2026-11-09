"""命令行入口：update / analyze / backtest / serve。"""

from __future__ import annotations

import time
from pathlib import Path

import typer

from .config import MARKET_NAMES, MARKETS, Paths, Watchlist, load_universe, normalize_code

app = typer.Typer(help="威克夫量价分析工具", no_args_is_help=True, add_completion=False)


def _parse_symbols(symbols: list[str]) -> list[tuple[str, str]]:
    """把 NVDA、BTC/USDT、crypto:BTC 之类的输入解析成 (market, code)。"""
    out = []
    for s in symbols:
        if ":" in s:
            m, c = s.split(":", 1)
        elif "/" in s:
            m, c = "crypto", s
        else:
            m, c = "us", s
        if m not in MARKETS:
            raise typer.BadParameter(f"不认识的市场 {m}")
        out.append((m, normalize_code(m, c)))  # type: ignore[arg-type]
    return out


@app.command()
def update(
    symbols: list[str] = typer.Argument(None, help="标的，如 NVDA BTC/USDT；不填则更新自选清单"),
    universe: bool = typer.Option(False, "--universe", help="同时更新回测标的池"),
    force: bool = typer.Option(False, "--force", help="全量重拉"),
):
    """增量拉取数据到本地缓存。"""
    from .data import DataService

    paths = Paths()
    svc = DataService(paths)
    items = _parse_symbols(symbols) if symbols else Watchlist(paths).items()
    starts = {m: load_universe(paths)[m]["start"] for m in MARKETS}
    if universe:
        uni = load_universe(paths)
        for m in MARKETS:
            for c in uni[m]["symbols"]:
                if (m, c) not in items:
                    items.append((m, c))
    t0 = time.time()
    fails = 0
    for k, (m, c) in enumerate(items, 1):
        try:
            df = svc.update(m, c, start=starts[m], force=force)
            typer.echo(f"[{k}/{len(items)}] {MARKET_NAMES[m]} {c:10} {len(df):5} 根，截至 {df.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            typer.secho(f"[{k}/{len(items)}] {MARKET_NAMES[m]} {c:10} 失败：{e}", fg="red")
    typer.echo(f"完成，用时 {time.time() - t0:.1f} 秒，失败 {fails} 个")


@app.command()
def analyze(
    symbols: list[str] = typer.Argument(None, help="标的，如 NVDA BTC/USDT"),
    all_: bool = typer.Option(False, "--all", help="分析整个自选清单"),
    refresh: bool = typer.Option(False, "--refresh", help="先更新数据"),
    out: Path = typer.Option(None, "--out", help="报告输出目录，默认 reports/"),
):
    """生成 HTML 报告：每个标的一份，加总览页和统计页。"""
    from .analysis import Analyzer
    from .report import render_reports

    paths = Paths()
    if all_ or not symbols:
        items = Watchlist(paths).items()
    else:
        items = _parse_symbols(symbols)
    if not items:
        typer.secho("没有要分析的标的", fg="yellow")
        raise typer.Exit(1)
    t0 = time.time()
    written = render_reports(Analyzer(paths), items, out_dir=out, refresh=refresh, progress=lambda c: typer.echo(f"分析 {c} …"))
    typer.echo(f"生成 {len(written)} 个文件到 {written[0].parent}，用时 {time.time() - t0:.1f} 秒")
    typer.echo(f"总览页：{written[-2]}")


@app.command()
def backtest(
    market: str = typer.Option("all", "--market", help="us / crypto / all"),
    refresh: bool = typer.Option(False, "--refresh", help="先更新标的池数据"),
    out: Path = typer.Option(None, "--out", help="统计文件路径，默认 data/stats.json"),
):
    """在回测标的池上重算每类事件的统计。"""
    from .backtest import load_stats, run_universe, save_stats
    from .config import load_universe
    from .data import DataService

    paths = Paths()
    svc = DataService(paths)
    markets = list(MARKETS) if market == "all" else [market]
    uni = load_universe(paths)
    existing = load_stats(paths.stats)
    stats_by, meta_by = {}, {}
    # 保留没重算的市场
    for m, mk in (existing.get("markets") or {}).items():
        if m not in markets:
            from .backtest.stats import GroupStats

            stats_by[m] = [GroupStats(**{**s, "horizons": {int(k): v for k, v in s["horizons"].items()}}) for s in mk["stats"]]
            meta_by[m] = mk["meta"]
    for m in markets:
        start = uni[m]["start"]

        def load_df(mkt, sym, _start=start):
            if refresh or not svc.has_cache(mkt, sym):
                return svc.update(mkt, sym, start=_start)
            return svc.read(mkt, sym)

        t0 = time.time()
        def progress(sym, k, n, _m=m):
            if k % 20 == 0 or k == n - 1:
                typer.echo(f"  {MARKET_NAMES[_m]} {k + 1}/{n} {sym}")

        stats, samples, meta = run_universe(m, load_df, paths=paths, progress=progress)
        stats_by[m], meta_by[m] = stats, meta
        typer.echo(f"{MARKET_NAMES[m]}：{len(meta['symbols'])} 个标的，{len(samples)} 个事件样本，用时 {time.time() - t0:.1f} 秒")
        for s in stats:
            h20 = s.horizons.get(20, {})
            typer.echo(
                f"  {s.group:12} n={s.n:4}  20日胜率 {h20.get('win_rate', float('nan')) * 100:5.1f}%  "
                f"超额 {h20.get('excess_mean', float('nan')) * 100:+5.2f}%  {s.verdict}"
            )
    path = out or paths.stats
    save_stats(path, stats_by, meta_by)
    typer.echo(f"统计写入 {path}")


@app.command()
def cases(
    tolerance: int = typer.Option(3, "--tolerance", help="命中容差，单位是 K 线根数"),
):
    """第 4 层验收：把 config/cases.yaml 里的经典案例和识别结果逐个对照。"""
    import pandas as pd
    import yaml

    from .analysis import Analyzer

    paths = Paths()
    raw = yaml.safe_load((paths.config / "cases.yaml").read_text(encoding="utf-8")) or {}
    an = Analyzer(paths)
    hits = total = 0
    for c in raw.get("cases", []):
        a = an.analyze(c["market"], c["symbol"])
        lo, hi = pd.Timestamp(c["window"][0]), pd.Timestamp(c["window"][1])
        idx = a.df.index
        found = [e for e in a.result.events if lo <= e.occurred <= hi]
        structs = [s for s in a.result.structures if lo <= s.climax_date <= hi and s.range_confirmed is not None]
        typer.secho(f"\n{c['name']}  [{c['symbol']} {c['window'][0]} ~ {c['window'][1]}]"
                    f"{'' if c.get('confirmed') else '  （人工日期待确认）'}", bold=True)
        if c["kind"] == "none":
            total += 1
            ok = not structs
            hits += ok
            typer.echo(f"  期望：无结构    识别：{len(structs)} 个已确认区间    {'命中' if ok else '未命中'}")
            for e in found:
                typer.echo(f"    - {e.kind} {e.occurred.date()}")
            continue
        for kind, day in (c.get("expected") or {}).items():
            total += 1
            target = pd.Timestamp(day)
            best = None
            for e in found:
                if e.kind != kind:
                    continue
                gap = abs(idx.get_indexer([e.occurred])[0] - idx.get_indexer([target], method="nearest")[0])
                if best is None or gap < best[0]:
                    best = (gap, e)
            if best and best[0] <= tolerance:
                hits += 1
                typer.echo(f"  {kind:6} 期望 {day}  识别 {best[1].occurred.date()}  相差 {best[0]} 根  命中")
            elif best:
                typer.echo(f"  {kind:6} 期望 {day}  识别 {best[1].occurred.date()}  相差 {best[0]} 根  未命中")
            else:
                typer.echo(f"  {kind:6} 期望 {day}  识别 —  未命中")
        others = [e for e in found if e.kind not in (c.get("expected") or {}) and e.kind != "RANGE"]
        if others:
            typer.echo("  另外识别出：" + "、".join(f"{e.kind} {e.occurred.date()}" for e in others))
    typer.secho(f"\n命中 {hits}/{total}（{hits / total:.0%}），验收标准 ≥ 80%", bold=True)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="只建议监听本机"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload", hidden=True),
):
    """启动本地 Web 服务。"""
    import uvicorn

    typer.echo(f"打开 http://{host}:{port}")
    if reload:
        uvicorn.run("wyckoff.web.app:create_app", host=host, port=port, factory=True, reload=True)
    else:
        from .web import create_app

        uvicorn.run(create_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
