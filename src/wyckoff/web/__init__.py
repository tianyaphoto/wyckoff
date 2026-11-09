"""本地 Web 服务：查看报告、管理自选清单。只监听 127.0.0.1。"""

from .app import create_app

__all__ = ["create_app"]
