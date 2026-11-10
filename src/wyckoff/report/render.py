"""模板渲染：静态报告和 Web 页面共用同一套模板。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..analysis import Analyzer, SymbolAnalysis
from ..backtest.stats import HORIZONS, MIN_SAMPLES
from ..config import MARKET_NAMES, MARKETS, Market, Paths
from ..detector.events import CONFIDENCE_ZH, EVENT_INFO, PHASE_ZH, STRUCTURE_ZH, event_name, phase_name
from .charts import figure_json
from .explain import EVENT_PLAIN, PHASE_STEPS, explain, row_plain, schematic_svg, stats_bottom_line

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"

DISCLAIMER = "历史统计不代表未来表现，本页面不构成投资建议。"


def fmt_pct(v, digits: int = 1, sign: bool = True) -> str:
    if v is None or v != v:
        return "—"
    s = f"{v * 100:+.{digits}f}%" if sign else f"{v * 100:.{digits}f}%"
    return s


def fmt_num(v, digits: int = 2) -> str:
    if v is None or v != v:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.{digits}f}"
    return f"{v:.{digits}f}"


def fmt_price(v) -> str:
    if v is None or v != v:
        return "—"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def make_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = fmt_pct
    env.filters["num"] = fmt_num
    env.filters["price"] = fmt_price
    env.globals.update(
        {
            "EVENT_INFO": EVENT_INFO,
            "PHASE_ZH": PHASE_ZH,
            "phase_name": phase_name,
            "event_name": event_name,
            "STRUCTURE_ZH": STRUCTURE_ZH,
            "CONFIDENCE_ZH": CONFIDENCE_ZH,
            "MARKET_NAMES": MARKET_NAMES,
            "MARKETS": MARKETS,
            "HORIZONS": HORIZONS,
            "MIN_SAMPLES": MIN_SAMPLES,
            "EVENT_PLAIN": EVENT_PLAIN,
            "PHASE_STEPS": PHASE_STEPS,
            "explain": explain,
            "row_plain": row_plain,
            "schematic_svg": schematic_svg,
            "stats_bottom_line": stats_bottom_line,
            "DISCLAIMER": DISCLAIMER,
            "VERSION": __version__,
            "CSS": (STATIC / "style.css").read_text(encoding="utf-8"),
        }
    )
    return env


class ReportRenderer:
    def __init__(self, analyzer: Analyzer, web: bool = False, link: Callable[[str, str], str] | None = None):
        self.analyzer = analyzer
        self.env = make_env()
        self.web = web
        self.link = link or (lambda m, c: f"./{m}_{c.replace('/', '_')}.html")

    def _ctx(self, **kw) -> dict:
        base = {
            "web": self.web,
            "link_symbol": self.link,
            "link_overview": "/" if self.web else "./index.html",
            "link_stats": "/stats" if self.web else "./stats.html",
            "link_watchlist": "/watchlist" if self.web else None,
            "link_guide": "/guide" if self.web else "./guide.html",
        }
        base.update(kw)
        return base

    def render(self, template: str, **kw) -> str:
        return self.env.get_template(template).render(**self._ctx(**kw))

    def symbol_page(self, a: SymbolAnalysis, **kw) -> str:
        return self.render("symbol.html", a=a, figure_json=figure_json(a), **kw)

    def overview_page(self, analyses: list[SymbolAnalysis], **kw) -> str:
        rows = [x.to_overview_row() for x in analyses]
        return self.render("overview.html", rows=rows, analyses=analyses, **kw)

    def guide_page(self, **kw) -> str:
        from ..config import load_params

        return self.render("guide.html", p=load_params("us", self.analyzer.paths), active="guide", **kw)

    def stats_page(self, **kw) -> str:
        return self.render("stats.html", stats=self.analyzer.stats_json(), **kw)


def render_reports(
    analyzer: Analyzer,
    items: list[tuple[Market, str]],
    out_dir: Path | None = None,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[Path]:
    """命令行入口：每个标的一份 HTML，加总览和统计页。"""
    paths = analyzer.paths
    out_dir = out_dir or paths.reports
    out_dir.mkdir(parents=True, exist_ok=True)
    r = ReportRenderer(analyzer, web=False)
    written: list[Path] = []
    analyses: list[SymbolAnalysis] = []
    for m, c in items:
        if progress:
            progress(c)
        a = analyzer.analyze(m, c, refresh=refresh)
        analyses.append(a)
        p = out_dir / f"{m}_{c.replace('/', '_')}.html"
        p.write_text(r.symbol_page(a), encoding="utf-8")
        written.append(p)
    idx = out_dir / "index.html"
    idx.write_text(r.overview_page(analyses), encoding="utf-8")
    written.append(idx)
    st = out_dir / "stats.html"
    st.write_text(r.stats_page(), encoding="utf-8")
    written.append(st)
    gd = out_dir / "guide.html"
    gd.write_text(r.guide_page(), encoding="utf-8")
    written.append(gd)
    return written
