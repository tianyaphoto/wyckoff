"""报告：plotly 图表 + Jinja2 模板，命令行和 Web 共用。"""

from .render import ReportRenderer, render_reports

__all__ = ["ReportRenderer", "render_reports"]
