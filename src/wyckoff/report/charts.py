"""K 线图：蜡烛 + 成交量 + 区间框 + 事件标记。"""

from __future__ import annotations



import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis import SymbolAnalysis
from ..detector.events import EVENT_INFO

ACC = "#1F8A5B"
DIST = "#C0392B"
UP = "#2E7D5B"
DOWN = "#C9463D"
INK = "#1B2430"
MUTED = "#6B7684"
GRID = "#E9EDF2"
ACCENT = "#1F5FBF"

MARKER = {
    "SC": dict(symbol="triangle-up", color=ACC, y="low", dy=-1.2),
    "BC": dict(symbol="triangle-down", color=DIST, y="high", dy=1.2),
    "AR": dict(symbol="circle", color=MUTED, y=None, dy=1.0),
    "SPRING": dict(symbol="star", color=ACC, y="low", dy=-1.2),
    "UTAD": dict(symbol="star", color=DIST, y="high", dy=1.2),
    "SOS": dict(symbol="arrow-up", color=ACC, y="high", dy=1.2),
    "SOW": dict(symbol="arrow-down", color=DIST, y="low", dy=-1.2),
}


def build_figure(a: SymbolAnalysis, bars: int = 320) -> go.Figure:
    df = a.df
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.76, 0.24]
    )
    if df.empty:
        fig.update_layout(title="没有数据")
        return fig
    x = df.index
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K 线",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
            increasing_fillcolor=UP,
            decreasing_fillcolor=DOWN,
            line_width=1,
            whiskerwidth=0.4,
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    frame = a.result.frame
    if "sma_trend" in frame:
        fig.add_trace(
            go.Scatter(
                x=x, y=frame["sma_trend"], mode="lines", name=f"{a.params.trend_sma} 日均线",
                line=dict(color=ACCENT, width=1.1), hoverinfo="skip",
            ),
            row=1, col=1,
        )
    vol_colors = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=x, y=df["volume"], marker_color=vol_colors, marker_opacity=0.55, name="成交量", showlegend=False,
               hovertemplate="%{x|%Y-%m-%d}<br>成交量 %{y:,.0f}<extra></extra>"),
        row=2, col=1,
    )
    if "vol_ratio" in frame:
        avg = df["volume"] / frame["vol_ratio"]
        fig.add_trace(
            go.Scatter(x=x, y=avg, mode="lines", name=f"{a.params.volume_period} 日均量",
                       line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"),
            row=2, col=1,
        )

    # 区间框
    for s in a.result.structures:
        if s.lower is None or s.ar_confirmed is None:
            continue
        color = ACC if s.kind == "accumulation" else DIST
        x0 = s.climax_date
        x1 = s.end if s.end is not None else df.index[-1]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=s.lower, y1=s.upper, xref="x", yref="y",
            line=dict(color=color, width=1, dash="dot"), fillcolor=color, opacity=0.08, layer="below",
        )
        # 标签放在区间框右下角内侧，避开画在高潮日的事件标记
        fig.add_annotation(
            x=x1, y=s.lower, xref="x", yref="y", text=f"{s.zh}区间", showarrow=False,
            xanchor="right", yanchor="bottom", xshift=-4, yshift=2, font=dict(size=10.5, color=color), opacity=0.85,
        )

    # 事件标记：画在发生日，悬停显示确认日和规则
    atr = frame["atr"] if "atr" in frame else None
    by_kind: dict[str, dict] = {}
    for e in a.result.events:
        if e.kind == "RANGE":
            continue
        m = MARKER[e.kind]
        row = df.loc[e.occurred]
        a_val = float(atr.loc[e.occurred]) if atr is not None and not pd.isna(atr.loc[e.occurred]) else 0.0
        if m["y"] == "low":
            y = row["low"] + m["dy"] * a_val
        elif m["y"] == "high":
            y = row["high"] + m["dy"] * a_val
        else:
            y = (row["high"] + m["dy"] * a_val) if e.structure == "accumulation" else (row["low"] - m["dy"] * a_val)
        lag = (e.confirmed - e.occurred).days
        hover = (
            f"<b>{e.label}</b><br>发生 {e.occurred.date()}　确认 {e.confirmed.date()}"
            + (f"（{lag} 天后）" if lag else "（当天）")
            + f"<br>价格 {e.price:,.2f}"
            + (f"　量比 {e.vol_ratio:.2f}" if e.vol_ratio is not None else "")
            + (f"　{e.volume_class}" if e.volume_class else "")
            + f"<br><span style='color:{MUTED}'>{e.rule}</span>"
        )
        d = by_kind.setdefault(e.kind, {"x": [], "y": [], "text": [], "hover": []})
        d["x"].append(e.occurred)
        d["y"].append(y)
        d["text"].append(e.kind)
        d["hover"].append(hover)
    for kind, d in by_kind.items():
        m = MARKER[kind]
        fig.add_trace(
            go.Scatter(
                x=d["x"], y=d["y"], mode="markers+text", name=f"{kind} {EVENT_INFO[kind]['zh']}",
                text=d["text"], textposition="top center" if m["dy"] > 0 else "bottom center",
                textfont=dict(size=10, color=m["color"]),
                marker=dict(symbol=m["symbol"], size=11, color=m["color"], line=dict(width=1, color="white")),
                hovertext=d["hover"], hoverinfo="text",
            ),
            row=1, col=1,
        )

    # 候选高潮：已经出现但还没等到 AR 确认，用空心标记画出来
    cand = candidate_point(a)
    if cand is not None:
        kind, day = cand
        m = MARKER[kind]
        a_val = float(atr.loc[day]) if atr is not None and not pd.isna(atr.loc[day]) else 0.0
        y = df.loc[day, "low"] - 1.2 * a_val if kind == "SC" else df.loc[day, "high"] + 1.2 * a_val
        fig.add_trace(
            go.Scatter(
                x=[day], y=[y], mode="markers+text", name=f"候选 {kind}（未确认）", text=[f"候选 {kind}"],
                textposition="bottom center" if kind == "SC" else "top center",
                textfont=dict(size=10, color=m["color"]),
                marker=dict(symbol=m["symbol"] + "-open", size=12, color=m["color"], line=dict(width=2, color=m["color"])),
                hovertext=[
                    f"<b>候选 {kind}（{EVENT_INFO[kind]['zh']}）</b><br>{day.date()}<br>"
                    f"<span style='color:{MUTED}'>振幅和成交量已达到高潮日标准，但还没等到随后的反向走势（AR），"
                    f"所以还不算确认，可能被否定</span>"
                ],
                hoverinfo="text",
            ),
            row=1, col=1,
        )

    # 默认显示最近 bars 根
    start = df.index[max(0, len(df) - bars)]
    visible = df.loc[start:]
    fig.update_yaxes(range=[0, float(visible["volume"].max()) * 1.1], row=2, col=1)
    pad = (visible["high"].max() - visible["low"].min()) * 0.14
    fig.update_xaxes(range=[start, df.index[-1] + pd.Timedelta(days=12)], rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(range=[start, df.index[-1] + pd.Timedelta(days=12)], row=2, col=1)
    fig.update_yaxes(range=[visible["low"].min() - pad, visible["high"].max() + pad], row=1, col=1)
    fig.update_layout(
        height=620,
        margin=dict(l=56, r=24, t=16, b=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font=dict(family="Inter, PingFang SC, Noto Sans SC, Helvetica Neue, Arial, sans-serif", color=INK, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor=GRID, showspikes=True, spikemode="across", spikethickness=1, spikecolor=MUTED),
        xaxis2=dict(showgrid=True, gridcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, tickformat=",.5~g", side="left", fixedrange=False),
        yaxis2=dict(showgrid=False, tickformat="~s"),
        dragmode="pan",
    )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] if a.market == "us" else [])
    return fig


def candidate_point(a: SymbolAnalysis) -> tuple[str, pd.Timestamp] | None:
    """当前候选高潮的类型和日期。日期取候选状态持续期间的最低点（SC）或最高点（BC）。"""
    kind = a.current.get("candidate")
    ph = a.result.phases
    if not kind or ph.empty or "candidate" not in ph:
        return None
    is_cand = (ph["candidate"] == kind).to_numpy()
    i = len(is_cand)
    while i > 0 and is_cand[i - 1]:
        i -= 1
    run = a.df.iloc[i:]
    if run.empty:
        return None
    return kind, (run["low"].idxmin() if kind == "SC" else run["high"].idxmax())


def chart_note(a: SymbolAnalysis, bars: int = 320) -> str | None:
    """默认视野里没有任何已确认事件时，告诉读者为什么图上看不到标记。"""
    df = a.df
    if df.empty:
        return None
    start = df.index[max(0, len(df) - bars)]
    events = [e for e in a.result.events if e.kind != "RANGE"]
    if any(e.occurred >= start for e in events):
        return None
    cand = candidate_point(a)
    tail = f"图上的空心标记是 {cand[1].date()} 的候选 {cand[0]}，它还没有被确认。" if cand else ""
    if not events:
        return f"这个标的历史上还没有识别出任何事件，所以图上没有标记。{tail}"
    last = events[-1]
    return (
        f"图上默认显示最近 {bars} 根 K 线（{start.date()} 起），这段时间没有已确认的事件，所以看不到标记。"
        f"最近一次是 {last.occurred.date()} 的 {last.label}。{tail}点上方的“3 年”或“全部”可以看到更早的标记和区间。"
    )


def figure_json(a: SymbolAnalysis) -> str:
    return build_figure(a).to_json()
