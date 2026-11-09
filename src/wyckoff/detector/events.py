"""事件和结构的数据模型，以及术语表。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

# kind -> (中文名, 一句话含义, 所属结构, 交易方向 +1 多 / -1 空)
EVENT_INFO: dict[str, dict] = {
    "RANGE": {"zh": "交易区间", "desc": "主力吸筹或派发的横盘范围", "side": 0},
    "SC": {"zh": "抛售高潮", "desc": "恐慌性抛售，大资金接盘", "side": +1},
    "AR": {"zh": "自动反弹/回落", "desc": "高潮后动能耗尽的第一波反向运动", "side": 0},
    "SPRING": {"name": "Spring", "zh": "弹簧/假跌破", "desc": "假跌破区间下沿，洗掉最后一批浮筹", "side": +1},
    "SOS": {"zh": "强势突破", "desc": "需求压倒供应，向上走出区间", "side": +1},
    "BC": {"zh": "买入高潮", "desc": "注意力最集中时的抢购，大资金出货", "side": -1},
    "UTAD": {"zh": "派发后的假突破", "desc": "假突破区间上沿，诱多后下跌", "side": -1},
    "SOW": {"zh": "弱势跌破", "desc": "供应压倒需求，向下走出区间", "side": -1},
}

# 区间里 AR 的方向取决于结构类型，回测时分开统计
AR_SIDE = {"accumulation": +1, "distribution": -1}

STRUCTURE_ZH = {"accumulation": "吸筹", "distribution": "派发"}
PHASE_ZH = {"A": "止跌/止涨", "B": "区间震荡", "C": "测试", "D": "突破", "E": "趋势展开"}


def phase_name(kind: str | None, phase: str | None) -> str | None:
    """阶段的中文名，A 阶段按结构类型区分止跌和止涨。"""
    if not phase:
        return None
    if phase == "A":
        return "止跌" if kind == "accumulation" else ("止涨" if kind == "distribution" else PHASE_ZH["A"])
    return PHASE_ZH[phase]
CONFIDENCE_ZH = {"high": "高", "medium": "中", "low": "低"}


def event_name(kind: str) -> str:
    """显示用的英文名：Spring 用习惯写法，其余是缩写。"""
    return EVENT_INFO[kind].get("name", kind)


def event_label(kind: str) -> str:
    return f"{event_name(kind)}（{EVENT_INFO[kind]['zh']}）"


def event_side(kind: str, structure: str) -> int:
    if kind == "AR":
        return AR_SIDE[structure]
    return EVENT_INFO[kind]["side"]


@dataclass
class Event:
    kind: str
    structure: str  # accumulation / distribution
    structure_id: int
    occurred: pd.Timestamp
    confirmed: pd.Timestamp
    price: float
    atr: float
    vol_ratio: float | None
    rule: str  # 触发规则的中文说明
    volume_class: str | None = None  # Spring/UTAD：缩量 / 放量

    @property
    def label(self) -> str:
        return event_label(self.kind)

    @property
    def side(self) -> int:
        return event_side(self.kind, self.structure)

    @property
    def lag_bars(self) -> int | None:
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["occurred"] = self.occurred.date().isoformat()
        d["confirmed"] = self.confirmed.date().isoformat()
        d["label"] = self.label
        d["side"] = self.side
        return d


@dataclass
class Structure:
    id: int
    kind: str  # accumulation / distribution
    climax_date: pd.Timestamp
    climax_price: float
    ar_date: pd.Timestamp | None = None
    ar_price: float | None = None
    lower: float | None = None
    upper: float | None = None
    range_atr: float | None = None
    ar_confirmed: pd.Timestamp | None = None
    range_confirmed: pd.Timestamp | None = None
    test_confirmed: pd.Timestamp | None = None
    breakout: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    end_reason: str | None = None  # breakout / failed / abandoned / breakout_failed / new_climax / ongoing
    events: list[Event] = field(default_factory=list)

    @property
    def zh(self) -> str:
        return STRUCTURE_ZH[self.kind]

    def to_dict(self) -> dict:
        def d(ts):
            return ts.date().isoformat() if ts is not None else None

        return {
            "id": self.id,
            "kind": self.kind,
            "kind_zh": self.zh,
            "climax_date": d(self.climax_date),
            "climax_price": self.climax_price,
            "ar_date": d(self.ar_date),
            "ar_price": self.ar_price,
            "lower": self.lower,
            "upper": self.upper,
            "range_atr": self.range_atr,
            "ar_confirmed": d(self.ar_confirmed),
            "range_confirmed": d(self.range_confirmed),
            "test_confirmed": d(self.test_confirmed),
            "breakout": d(self.breakout),
            "end": d(self.end),
            "end_reason": self.end_reason,
        }
