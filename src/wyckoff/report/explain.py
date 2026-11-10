"""把识别结果翻译成大白话：现在是什么情况、接下来看什么、什么情况下这个判断作废。

页面上所有“解释性”的文字都从这里出，模板只负责摆放。
"""

from __future__ import annotations

from markupsafe import Markup

from ..analysis import SymbolAnalysis

# 每个事件的大白话解释：长什么样、理论上说明什么
EVENT_PLAIN: dict[str, dict[str, str]] = {
    "SC": {
        "look": "下跌途中突然出现一根特别长、成交量特别大的 K 线，并且创了近期新低。",
        "mean": "理论上这是恐慌盘集中卖出、大资金开始接货的一天，下跌可能告一段落。",
    },
    "BC": {
        "look": "上涨途中突然出现一根特别长、成交量特别大的 K 线，并且创了近期新高。",
        "mean": "理论上这是追涨情绪最热、大资金开始出货的一天，上涨可能告一段落。",
    },
    "AR": {
        "look": "高潮之后几天内出现的第一波明显反向走势：暴跌后的反弹，或暴涨后的回落。",
        "mean": "它和高潮日一起划出交易区间的上下沿。高潮日要等 AR 出现才算确认。",
    },
    "RANGE": {
        "look": "AR 之后价格在高潮价和 AR 价之间来回走了至少 15 根 K 线。",
        "mean": "理论上大资金在这个范围里慢慢买（吸筹）或慢慢卖（派发）。图上用虚线框标出。",
    },
    "SPRING": {
        "look": "价格短暂跌破区间下沿，几天内又收回到区间里。",
        "mean": "理论上是最后一次洗盘：把止损盘震出去后再涨。跌破时成交量越小越符合理论。",
    },
    "UTAD": {
        "look": "价格短暂涨破区间上沿，几天内又跌回区间里。",
        "mean": "理论上是诱多：把追高的人套住后再跌。它是 Spring 的镜像。",
    },
    "SOS": {
        "look": "收盘放量站上区间上沿。",
        "mean": "理论上说明买方已经压过卖方，区间结束，进入上涨。",
    },
    "SOW": {
        "look": "收盘放量跌破区间下沿。",
        "mean": "理论上说明卖方已经压过买方，区间结束，进入下跌。",
    },
}

PHASE_STEPS = [
    ("A", "止跌 / 止涨", "出现高潮日和第一波反向走势，原来的趋势被打断。"),
    ("B", "区间震荡", "价格在上下沿之间来回，这是时间最长的一段。"),
    ("C", "测试", "出现一次假跌破（Spring）或假突破（UTAD），随后收回。"),
    ("D", "突破", "放量走出区间，方向确定。"),
    ("E", "趋势展开", "突破后超过 20 根 K 线，新趋势在区间之外运行。"),
]

CAVEAT = "这是对走势形态的描述，不是买卖建议。回测显示这些信号在历史上没有稳定跑赢“随便哪天买”。"


def _price(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:,.2f}"
    return f"{v:.4f}"


def explain(a: SymbolAnalysis) -> dict:
    """返回 {title, now, watch, invalid, steps, caveat}，全部是完整的中文句子。"""
    cur = a.current
    kind, phase = cur.get("kind"), cur.get("phase")
    lo, up = cur.get("lower"), cur.get("upper")
    p = a.params
    out = {"caveat": CAVEAT, "steps": [], "phase": phase}

    if not phase:
        trend = cur.get("trend") or "震荡"
        trend_text = {
            "上升趋势": "50 日均线在往上走，价格处在上升趋势里",
            "下降趋势": "50 日均线在往下走，价格处在下降趋势里",
            "震荡": "50 日均线基本走平，价格没有明显方向",
        }.get(trend, trend)
        out["title"] = "现在没有可分析的结构"
        out["now"] = (
            f"{a.code} 最近没有出现“放量暴跌”或“放量暴涨”式的高潮日，所以工具没有画出交易区间。"
            f"目前 {trend_text}。大多数标的在大多数时候都是这个状态，这很正常。"
        )
        out["watch"] = (
            f"工具在等一个高潮日：当天振幅至少是平时的 {p.climax_range_atr:g} 倍、成交量至少是平时的 "
            f"{p.climax_volume_ratio:g} 倍，并且创 {p.extreme_lookback} 日新低或新高。出现之后才会开始跟踪。"
        )
        out["invalid"] = "没有正在跟踪的判断，所以也没有什么会作废。"
        return out

    acc = kind == "accumulation"
    side = "吸筹" if acc else "派发"
    climax = "SC" if acc else "BC"
    test = "Spring" if acc else "UTAD"
    brk = "SOS" if acc else "SOW"
    climax_edge, ar_edge = ("下沿", "上沿") if acc else ("上沿", "下沿")
    go, back = ("涨", "跌") if acc else ("跌", "涨")
    rng = f"{_price(lo)} 到 {_price(up)}" if lo is not None and up is not None else "高潮价和 AR 价之间"
    out["steps"] = [(k, (("止跌" if acc else "止涨") if k == "A" else name), k == phase) for k, name, _ in PHASE_STEPS]
    out["title"] = f"疑似{side}，走到第 {'ABCDE'.index(phase) + 1} 步（共 5 步）"

    theory = (
        "威克夫理论认为，大跌之后大资金会在一个横盘区间里慢慢买入，买够了再拉升。"
        if acc
        else "威克夫理论认为，大涨之后大资金会在一个横盘区间里慢慢卖出，卖完了价格再下跌。"
    )
    if phase == "A":
        if cur.get("candidate"):
            now = (
                f"{a.code} 刚出现一根符合{climax}（{'抛售' if acc else '买入'}高潮）条件的 K 线，"
                f"但还没有等到随后的反向走势（AR），所以只是候选，随时可能被否定。"
            )
            watch = f"接下来 {p.ar_window} 根 K 线内有没有明显的反{'弹' if acc else '向回落'}。有，高潮日才算确认。"
            invalid = f"{p.ar_max_bars} 根 K 线内没有等到 AR，这个候选会被放弃。"
        else:
            now = (
                f"{a.code} 出现了{climax}（放量{'暴跌' if acc else '暴涨'}）和随后的 AR（第一波{'反弹' if acc else '回落'}），"
                f"原来的{'下跌' if acc else '上涨'}被打断了。这两天的价格划出了一个可能的区间：{rng}。"
            )
            watch = (
                f"价格能不能在这个范围里待满 {p.range_min_bars} 根 K 线。待满了，工具才确认这是一个交易区间，进入 B 阶段。"
            )
            invalid = f"收盘连续 {p.range_fail_bars} 天明显{'跌破' if acc else '涨破'}{climax_edge}，说明原来的趋势还在继续，这个结构作废。"
    elif phase == "B":
        now = (
            f"{a.code} 在 {rng} 之间横盘。{theory}现在处在最长也最无聊的一段：区间震荡。"
            f"只凭这一步还看不出最后往哪边走。"
        )
        watch = (
            f"两件事：一是出现{test}（短暂{'跌破' if acc else '涨破'}{climax_edge}后收回），"
            f"二是直接放量{'站上' if acc else '跌破'}{ar_edge}（{brk}）。"
        )
        invalid = f"收盘连续 {p.range_fail_bars} 天远离区间（超出 {p.range_fail_atr:g} 倍日均波动）而且没放量，区间算失败。"
    elif phase == "C":
        now = (
            f"{a.code} 在 {rng} 的区间里出现过{test}：价格短暂{'跌破' if acc else '涨破'}{climax_edge}后又收了回来。"
            f"理论上这是{'最后一次洗盘' if acc else '最后一次诱多'}，是五步里证据最多的一步。"
        )
        watch = f"能不能放量{'站上' if acc else '跌破'}{ar_edge} {_price(up if acc else lo)}（{brk}）。出现就进入 D 阶段。"
        invalid = f"价格再次{'跌破' if acc else '涨破'}{climax_edge}并且收不回来，区间算失败。"
    elif phase == "D":
        now = (
            f"{a.code} 已经放量{'站上' if acc else '跌破'}区间{ar_edge}（{brk}），区间 {rng} 结束，方向是往{go}。"
            f"这是突破后的头 {p.phase_d_bars} 根 K 线。"
        )
        watch = f"突破后能不能守在区间外面。常见的走法是回头碰一下{ar_edge}再继续{go}。"
        invalid = f"收盘{back}回区间里面，算突破失败。"
    else:
        now = (
            f"{a.code} 走出 {rng} 的区间已经超过 {p.phase_d_bars} 根 K 线，新趋势在区间外运行。"
            f"这个{side}结构的五步已经走完。"
        )
        watch = "这一步没有新的威克夫事件可等。工具会继续跟踪，直到出现下一个高潮日。"
        invalid = f"收盘{back}回原来的区间，或者出现新的高潮日，这个结构就结束。"

    conf = cur.get("confidence")
    conf_text = {
        "high": "可信度高：高潮、AR 和测试三样都已确认。",
        "medium": f"可信度中：高潮和 AR 已确认，区间也满了 {p.range_min_bars} 根，但还没出现测试。",
        "low": "可信度低：证据还少，结构随时可能被否定。",
    }.get(conf, "")
    out.update(now=now, watch=watch, invalid=invalid, confidence=conf_text)
    return out


def row_plain(row: dict) -> str:
    """总览表里的一句话解释。"""
    kind, phase = row.get("kind"), row.get("phase")
    if not phase:
        return {
            "上升趋势": "没有横盘区间，价格在上升趋势里",
            "下降趋势": "没有横盘区间，价格在下降趋势里",
            "震荡": "没有横盘区间，也没有明显方向",
        }.get(row.get("trend") or "", "数据不足，暂时无法判断")
    acc = kind == "accumulation"
    if phase == "A" and row.get("candidate"):
        return "刚出现放量暴跌，还要等反弹来确认" if acc else "刚出现放量暴涨，还要等回落来确认"
    return {
        "A": "刚放量暴跌并反弹，等它横住" if acc else "刚放量暴涨并回落，等它横住",
        "B": "大跌后在区间里横盘，方向未定" if acc else "大涨后在区间里横盘，方向未定",
        "C": "出现过假跌破并收回，等向上突破" if acc else "出现过假突破并跌回，等向下跌破",
        "D": "已放量向上突破区间" if acc else "已放量向下跌破区间",
        "E": "突破后上涨趋势在延续" if acc else "跌破后下跌趋势在延续",
    }[phase]


def stats_bottom_line(market_stats: list[dict]) -> str:
    """统计页每个市场的一句话结论。"""
    sig = [s["group"] for s in market_stats if s.get("verdict") == "显著优于基准"]
    if not sig:
        return "一句话结论：这个市场里没有任何一类事件在 20 天持有期上稳定跑赢“随便哪天入场”。"
    return (
        f"一句话结论：只有 {'、'.join(sig)} 在 20 天持有期上跑赢了基准。"
        "分组一多，碰巧显著的概率也会变大，样本外未必还成立。"
    )


# ---------------------------------------------------------------- 示意图


def schematic_svg(kind: str = "accumulation") -> Markup:
    """吸筹（或镜像后的派发）五个阶段的示意图。颜色用页面的 CSS 变量。"""
    acc = kind == "accumulation"
    H = 310

    def y(v: float) -> float:
        return v if acc else H - v

    pts = [(20, 40), (95, 150), (150, 235), (205, 110), (255, 205), (305, 125), (355, 195), (405, 135),
           (455, 262), (492, 170), (520, 150), (565, 70), (605, 108), (700, 30)]
    path = " ".join(f"{x},{y(v):.0f}" for x, v in pts)
    color = "var(--acc)" if acc else "var(--dist)"
    top, bottom = sorted((y(110), y(235)))
    labels = [
        (150, 235, "SC" if acc else "BC", "放量暴跌" if acc else "放量暴涨", +1),
        (205, 110, "AR", "第一波反弹" if acc else "第一波回落", -1),
        (455, 262, "Spring" if acc else "UTAD", "假跌破后收回" if acc else "假突破后跌回", +1),
        (565, 70, "SOS" if acc else "SOW", "放量突破" if acc else "放量跌破", -1),
    ]
    parts = [
        f'<svg viewBox="0 0 720 {H}" role="img" aria-label="{"吸筹" if acc else "派发"}五个阶段示意图" class="schematic">',
        f'<rect x="150" y="{top:.0f}" width="390" height="{bottom - top:.0f}" fill="{color}" fill-opacity=".06" '
        f'stroke="{color}" stroke-dasharray="5 4" stroke-opacity=".7"/>',
        f'<text x="360" y="{top - 7:.0f}" text-anchor="middle" class="sch-range">交易区间</text>',
        f'<polyline points="{path}" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linejoin="round"/>',
    ]
    for x, v, name, desc, below in labels:
        yy = y(v)
        d = below if acc else -below  # 标签放在折线外侧
        parts.append(f'<circle cx="{x}" cy="{yy:.0f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x}" y="{yy + (20 if d > 0 else -22):.0f}" text-anchor="middle" class="sch-name" fill="{color}">{name}</text>')
        parts.append(f'<text x="{x}" y="{yy + (34 if d > 0 else -9):.0f}" text-anchor="middle" class="sch-desc">{desc}</text>')
    bands = [("A", 150, 255), ("B", 255, 425), ("C", 425, 492), ("D", 492, 610), ("E", 610, 700)]
    for name, x0, x1 in bands:
        parts.append(f'<line x1="{x0}" y1="12" x2="{x0}" y2="{H - 8}" stroke="var(--line-2)" stroke-dasharray="2 4"/>')
        parts.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{22 if acc else H - 6}" text-anchor="middle" class="sch-phase">{name}</text>')
    parts.append("</svg>")
    return Markup("".join(parts))
