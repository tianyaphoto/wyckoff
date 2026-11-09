"""后台任务：拉取数据并分析。状态存 SQLite，页面轮询。"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

STATUS_ZH = {"queued": "排队中", "fetching": "拉取数据", "analyzing": "分析中", "done": "完成", "failed": "失败"}
STEPS = ["queued", "fetching", "analyzing", "done"]


@dataclass
class Task:
    id: str
    kind: str  # add / update
    market: str
    code: str
    status: str
    message: str
    created: float
    updated: float

    @property
    def status_zh(self) -> str:
        return STATUS_ZH.get(self.status, self.status)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "fetching", "analyzing")

    @property
    def step_index(self) -> int:
        return STEPS.index(self.status) if self.status in STEPS else (3 if self.status == "failed" else 0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "market": self.market,
            "code": self.code,
            "status": self.status,
            "status_zh": self.status_zh,
            "message": self.message,
            "active": self.active,
        }


class TaskStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, kind TEXT, market TEXT, code TEXT, "
                "status TEXT, message TEXT, created REAL, updated REAL)"
            )
            # 上次进程退出时还在跑的任务，标记失败
            c.execute(
                "UPDATE tasks SET status='failed', message='服务重启，任务中断，请重试' WHERE status IN ('queued','fetching','analyzing')"
            )

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def create(self, kind: str, market: str, code: str) -> Task:
        now = time.time()
        t = Task(uuid.uuid4().hex[:12], kind, market, code, "queued", "", now, now)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?)",
                (t.id, t.kind, t.market, t.code, t.status, t.message, t.created, t.updated),
            )
        return t

    def set_status(self, task_id: str, status: str, message: str = "") -> None:
        with self._lock, self._conn() as c:
            c.execute("UPDATE tasks SET status=?, message=?, updated=? WHERE id=?", (status, message, time.time(), task_id))

    def get(self, task_id: str) -> Task | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return Task(*row) if row else None

    def recent(self, limit: int = 20, symbol: tuple[str, str] | None = None, since: float | None = None) -> list[Task]:
        q = "SELECT * FROM tasks"
        args: list = []
        conds = []
        if symbol:
            conds.append("market=? AND code=?")
            args += list(symbol)
        if since is not None:
            conds.append("updated >= ?")
            args.append(since)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [Task(*r) for r in rows]

    def active_or_recent(self, symbol: tuple[str, str] | None = None, window_s: float = 30.0) -> list[Task]:
        """进行中的任务，加上最近 window_s 秒内结束的任务，供页面显示。"""
        tasks = self.recent(limit=30, symbol=symbol)
        now = time.time()
        return [t for t in tasks if t.active or now - t.updated <= window_s]

    def has_active(self, market: str, code: str) -> bool:
        return any(t.active for t in self.recent(limit=10, symbol=(market, code)))


class TaskRunner:
    """两个工作线程，顺序执行任务。"""

    def __init__(self, store: TaskStore, work: Callable[[Task, Callable[[str, str], None]], None], workers: int = 2):
        self.store = store
        self.work = work
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wyckoff-task")

    def submit(self, kind: str, market: str, code: str) -> Task:
        t = self.store.create(kind, market, code)
        self.pool.submit(self._run, t)
        return t

    def _run(self, t: Task) -> None:
        def set_status(status: str, message: str = ""):
            self.store.set_status(t.id, status, message)

        try:
            self.work(t, set_status)
            set_status("done", "完成")
        except Exception as e:  # noqa: BLE001
            set_status("failed", str(e) or e.__class__.__name__)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
