# 威克夫量价分析工具

识别美股和加密货币日线上的威克夫阶段与关键事件（SC、AR、Spring、SOS、BC、UTAD、SOW），给每类事件附上回测标的池上的历史统计。不预测具体价格，不自动下单。

设计文档：<https://claude.ai/artifact/NiJT523SZGsMMtGz4ytHNk>

## 安装

需要 Python 3.11 和 [uv](https://docs.astral.sh/uv/)。

```bash
cd ~/projects/wyckoff
uv sync
uv run playwright install chromium   # 只有跑端到端测试才需要
```

## 使用

```bash
uv run wyckoff update                  # 增量拉取自选清单的数据
uv run wyckoff update --universe       # 连回测标的池一起拉（首次约 2 分钟）
uv run wyckoff backtest                # 在标的池上重算事件统计 → data/stats.json
uv run wyckoff analyze --all           # 生成 HTML 报告 → reports/index.html
uv run wyckoff analyze NVDA BTC/USDT   # 只分析指定标的
uv run wyckoff serve                   # 本地 Web 服务 → http://127.0.0.1:8000
```

Web 页面能看总览、标的详情、事件统计，还能新增和删除自选标的。第一次用先看「新手说明」页（`/guide`），每个页面也都带大白话解读。它只监听本机，没有登录。

离线演示：`WYCKOFF_FAKE_DATA=1 uv run wyckoff serve` 会用合成行情代替真实数据源。

## 配置

| 文件 | 内容 |
| --- | --- |
| `config/watchlist.yaml` | 自选清单，Web 页面和命令行都读写它 |
| `config/universe.yaml` | 回测标的池：标普 100 加 QQQ/SPY/IWM，币安 20 个老交易对 |
| `config/params.yaml` | 规则参数，全部以 ATR 倍数表示，可按市场覆盖 |

数据缓存在 `data/`（Parquet），报告在 `reports/`，两者都不提交到 git。

## 测试

```bash
uv run pytest                 # 单元和集成测试，约 5 秒，不联网
uv run pytest -m e2e          # Playwright 端到端，截图存到 reports/e2e/
uv run pytest -m network      # 需要本地已有真实数据缓存
```

测试分层见 `docs/acceptance.md`。

## 目录

```
src/wyckoff/
  config.py        路径、参数、自选清单、标的池
  data/            yfinance / 币安适配、Parquet 缓存、增量更新、合成数据源
  indicators.py    ATR、量比、均线、N 日极值
  detector/        逐日推进的状态机：区间、8 种事件、A–E 阶段
  backtest/        前瞻收益、基准、bootstrap 显著性
  analysis.py      把数据、识别、统计串起来
  report/          plotly 图表 + Jinja2 模板，命令行和 Web 共用；explain.py 负责大白话解读
  web/             FastAPI + HTMX，后台任务存 SQLite
  cli.py           typer 命令行
  synthetic.py     合成行情：随机游走、手搭场景、镜像
tests/             规格对照、因果性、统计校验、数据层、Web、端到端
```
